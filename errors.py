from collections import defaultdict
from pathlib import Path
import json
import jsonlines
import typing as tp
from enum import Enum
import numpy as np
from numpy import mean
from itertools import chain


class ErrorType(Enum):
    MISALIGNED_SPAN = 0,


EXTRA_SPAN = 1,
MISSING_SPAN = 2,
ENTITY_MERGED = 3,
ENTITY_SPLIT = 4,
MISSING_ENTITY = 5,
EXTRA_ENTITY = 6,


class GoldPredAlignment:
    def __init__(self, gold, pred, tokens):
        self.error_counts: tp.DefaultDict[ErrorType, int] = defaultdict(int)
        self.alignment = None
        self.errors: tp.DefaultDict[ErrorType, tp.List[tp.Tuple]] = defaultdict(list)
        self.tokens: tp.List[str] = tokens
        self.gold_markup: tp.List[tp.Tuple] = list(
            chain.from_iterable([[self.token_to_span(y, f"GOLD_{i}", ) for y in x] for i, x in enumerate(gold)]))
        self.pred_markup: tp.List[tp.Tuple] = list(
            chain.from_iterable([[self.token_to_span(y, f"PRED_{i}", ) for y in x] for i, x in enumerate(pred)]))
        self.gold_markup_errors_only: tp.List[tp.Tuple] = []
        self.pred_markup_errors_only: tp.List[tp.Tuple] = []
        self.collect_errors(gold, pred)

    def overlap_score(self, gold_span, pred_spans):
        overlapping = [i for i in pred_spans if gold_span[0] <= i[0] < gold_span[1] or gold_span[0] < i[1] <= gold_span[1]]
        # if not overlapping:
        # self.error_counts[ErrorType.MISSING_SPAN] += 1
        scores = []
        for span in overlapping:
            overlap_length = max(0, min(span[1], gold_span[1]) - max(span[0], gold_span[0]))
            # overlap_length = span[1] - span[0] + gold_span[1] - gold_span[0]
            gold_length = gold_span[1] - gold_span[0]
            span_length = span[1] - span[0]
            outside_length = span_length - overlap_length
            score = (overlap_length / gold_length) / (outside_length + 1)
            scores.append(score)

            return overlapping, scores

    @staticmethod
    def sort_clusters(clusters: tp.List):
        clusters = [list(map(lambda x: sorted(x, key=lambda y: y[0], reverse=True), cluster)) for cluster in clusters]
        return sorted(clusters, key=lambda x: x[0][0])

    def entity_alignment_score(self, gold: tp.List[tp.List[int]], pred: tp.List[tp.List[int]]):
        span_scores = []
        res_overlaps = []
        pred_overlaps = {x: False for x, _ in enumerate(pred)}
        has_error = False
        # add something like pred_span_has_overlaps: extra span
        for span in gold:
            # if not overlaps: missing span
            overlaps, overlap_scores = self.overlap_score(span, pred)
            for ix, npred in enumerate(pred):
                if npred in overlaps:
                    pred_overlaps[ix] = True
            # highest overlap not 1.0: misaligned span
            if overlap_scores and max(overlap_scores) != 1.0:
                self.error_counts[ErrorType.MISALIGNED_SPAN] += 1
                self.errors[ErrorType.MISALIGNED_SPAN].append((gold, overlaps[np.argmin(overlap_scores)]))
                has_error = True
            span_score = mean(overlap_scores) if overlap_scores else 0
            span_scores.append(span_score)
            res_overlaps.append((overlaps, overlap_scores,))

        if mean(span_scores):
            # this entire piece of logic has to be moved outside
            if not all(x[0] for x in res_overlaps):
                self.error_counts[ErrorType.MISSING_SPAN] += 1
                # has_error = True
            if not all(pred_overlaps.values()):
                self.error_counts[ErrorType.EXTRA_SPAN] += 1
                # has_error = True
        return mean(span_scores), res_overlaps, pred_overlaps

    def collect_errors(self, gold, pred):
        scores = defaultdict(list)
        gold_error_ix: set[int] = set()
        pred_error_ix: set[int] = set()
        overlaps_res = []
        overlaps_pred = []
        for ix, ge in enumerate(gold):
            for pix, pe in enumerate(pred):
                alignment_score, res_overlaps, pred_overlaps = self.entity_alignment_score(ge, pe)
                overlaps_res.append(res_overlaps)
                overlaps_pred.append(pred_overlaps)
                # if has_error:
                    # gold_error_ix.add(ix)
                    # pred_error_ix.add(pix)
                scores[ix].append(alignment_score)

        self.alignment = np.array([x[1] for x in sorted(scores.items(), key=lambda x: x[0])])

        # here find top match for every gold entity
        # also analyse the overlaps and find spans with no overlaps in res and pred
        
        nonzero_rows = np.count_nonzero(self.alignment, axis=0)
        if any(nonzero_rows > 1):
            self.error_counts[ErrorType.ENTITY_MERGED] = sum(nonzero_rows > 1)
            indices = list((nonzero_rows > 1).nonzero()[0])
            pred_error_ix.update(indices)
        if any(nonzero_rows == 0):
            self.error_counts[ErrorType.EXTRA_ENTITY] = sum(nonzero_rows == 0)
            indices = list((nonzero_rows == 1).nonzero()[0])
            pred_error_ix.update(indices)

        nonzero_columns = np.count_nonzero(self.alignment, axis=1)
        if any(nonzero_columns > 1):
            self.error_counts[ErrorType.ENTITY_SPLIT] = sum(nonzero_columns > 1)
            indices = list((nonzero_columns > 1).nonzero()[0])
            gold_error_ix.update(indices)
        if any(nonzero_columns == 0):
            self.error_counts[ErrorType.MISSING_ENTITY] = sum(nonzero_columns == 0)
            indices = list((nonzero_columns == 1).nonzero()[0])
            gold_error_ix.update(indices)

        self.gold_markup_errors_only = list(
            chain.from_iterable([[self.token_to_span(y, f"GOLD_{i}", ) for y in x] for x, i in [(gold[j], j,) for j in gold_error_ix]]))
        self.pred_markup_errors_only = list(
            chain.from_iterable([[self.token_to_span(y, f"PRED_{i}", ) for y in x] for x, i in [(pred[j], j,) for j in pred_error_ix]]))


def line(self) -> str:
    return json.dumps({"text": " ".join(self.tokens), "label": self.gold_markup + self.pred_markup})


def error_line(self) -> str:
    return json.dumps({"text": " ".join(self.tokens), "label": self.gold_markup_errors_only + self.pred_markup_errors_only})


def token_to_span(self, token, label):
    start = len(" ".join(self.tokens[0:token[0]])) + 1
    length = len(" ".join(self.tokens[token[0]:token[1]]))
    return start, start + length, label

# def link_entities(gold, pred, scores):


pred_path = Path("data/conll_logs/pred.json")
gold_path = Path("data/conll_logs/gold.json")

pred_sents = list(jsonlines.open(pred_path))
gold_sents = list(jsonlines.open(gold_path))

assert len(pred_sents) == len(gold_sents)

errors = {x: [] for x in ErrorType}
total_error_counts: tp.DefaultDict[ErrorType, int] = defaultdict(int)
for g, p in zip(gold_sents, pred_sents):
    if g["clusters"] and p["clusters"] and g["clusters"] != p["clusters"]:
        alignment = GoldPredAlignment(g["clusters"], p["clusters"], g["cased_words"])
        for key, value in alignment.error_counts.items():
            errors[key].append(alignment)
            total_error_counts[key] += value

for key, value in errors.items():
    with Path(f"data/conll_logs/{key}.json").open("w") as io:
        io.writelines([x.error_line() + "\n" for x in value])

# def align_entities(gold: tp.List[tp.List[int, int]], pred: tp.List[tp.List[int, int]]):
# for ix, entity in gold:
