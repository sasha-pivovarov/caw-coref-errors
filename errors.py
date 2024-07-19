import random
from collections import defaultdict
from pathlib import Path
import json
import jsonlines
import typing as tp
from enum import Enum
import numpy as np
from numpy import mean
from itertools import chain
from collections import Counter
from scipy.stats import chisquare


class ErrorType(Enum):
    MISALIGNED_SPAN = 0,
    EXTRA_SPAN = 1,
    MISSING_SPAN = 2,
    ENTITY_MERGED = 3,
    ENTITY_SPLIT = 4,
    MISSING_ENTITY = 5,
    EXTRA_ENTITY = 6,


class GoldPredAlignment:

    def __init__(self, gold, pred, tokens, deprel, head2span, tags):
        self.pred_error_ix = set()
        self.gold_error_ix = set()
        self.deprel = deprel
        self.head2span = head2span
        self.dict_head2span = {(x[1], x[2],): x[0] for x in head2span}
        self.tags = tags
        self.error_counts: tp.DefaultDict[ErrorType, int] = defaultdict(int)
        self.errors: tp.DefaultDict[ErrorType, tp.List[tp.Tuple]] = defaultdict(list)
        self.tokens: tp.List[str] = tokens
        self.gold_markup: tp.List[tp.Tuple] = list(chain.from_iterable([[self.token_to_span(y, f"GOLD_{i}",) for y in x] for i, x in enumerate(gold)]))
        self.pred_markup: tp.List[tp.Tuple] = list(chain.from_iterable([[self.token_to_span(y, f"PRED_{i}",) for y in x] for i, x in enumerate(pred)]))
        self.gold_markup_errors_only: tp.List[tp.Tuple] = []
        self.pred_markup_errors_only: tp.List[tp.Tuple] = []
        self.all_spans = list(chain.from_iterable(gold))
        self.gold = gold
        self.pred = pred
        self.missing_spans = []
        self.extra_spans = []
        self.alignment = None
        self.collect_errors()

    def overlap_score(self, gold_span, pred_spans):
        overlapping = [i for i in pred_spans if gold_span[0] <= i[0] < gold_span[1] or gold_span[0] < i[1] <= gold_span[1]]
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
            res_overlaps.append(overlaps)

        return mean(span_scores), res_overlaps, pred_overlaps

    def collect_errors(self):
        scores = defaultdict(list)
        overlaps_res = [[] for _ in range(len(self.gold))]
        overlaps_pred = [[] for _ in range(len(self.pred))]
        for ix, ge in enumerate(self.gold):
            for pix, pe in enumerate(self.pred):
                alignment_score, res_overlaps, pred_overlaps = self.entity_alignment_score(ge, pe)
                overlaps_res[ix].append(res_overlaps)
                overlaps_pred[pix].append(pred_overlaps)
                scores[ix].append(alignment_score)

        self.alignment = np.array([x[1] for x in sorted(scores.items(), key=lambda x: x[0])])

        # here find top match for every gold entity
        # also analyse the overlaps and find spans with no overlaps in res and pred
        self._find_extra_spans(overlaps_pred)
        self._find_missing_spans(overlaps_res)
        self.error_counts[ErrorType.MISSING_SPAN] = len(self.missing_spans)
        self.error_counts[ErrorType.EXTRA_SPAN] = len(self.extra_spans)
        nonzero_rows = np.count_nonzero(self.alignment, axis=0)

        if any(nonzero_rows > 1):
            self.error_counts[ErrorType.ENTITY_MERGED] = sum(nonzero_rows > 1)
            indices = list((nonzero_rows > 1).nonzero()[0])
            self.pred_error_ix.update(indices)
        if any(nonzero_rows == 0):
            self.error_counts[ErrorType.EXTRA_ENTITY] = sum(nonzero_rows == 0)
            indices = list((nonzero_rows == 1).nonzero()[0])
            self.pred_error_ix.update(indices)

        nonzero_columns = np.count_nonzero(self.alignment, axis=1)
        if any(nonzero_columns > 1):
            self.error_counts[ErrorType.ENTITY_SPLIT] = sum(nonzero_columns > 1)
            indices = list((nonzero_columns > 1).nonzero()[0])
            self.gold_error_ix.update(indices)
        if any(nonzero_columns == 0):
            self.error_counts[ErrorType.MISSING_ENTITY] = sum(nonzero_columns == 0)
            indices = list((nonzero_columns == 1).nonzero()[0])
            self.gold_error_ix.update(indices)

        self.gold_markup_errors_only = list(chain.from_iterable([[self.token_to_span(y, f"GOLD_{i}",) for y in x] for x, i in [(self.gold[j], j,) for j in
                                                                                                                               self.gold_error_ix]]))
        self.pred_markup_errors_only = list(chain.from_iterable([[self.token_to_span(y, f"PRED_{i}",) for y in x] for x, i in [(self.pred[j], j,) for j in
                                                                                                                               self.pred_error_ix]]))

    def line(self) -> str:
        return json.dumps({"text": " ".join(self.tokens), "label": self.gold_markup + self.pred_markup, "tags": self.tags, "deprel": self.deprel, "head2span": self.head2span})

    def error_line(self) -> str:
        return json.dumps({"text": " ".join(self.tokens), "label": self.gold_markup_errors_only + self.pred_markup_errors_only, "tags": self.tags, "deprel": self.deprel, "head2span": self.head2span, "tokens": self.tokens})

    def token_to_span(self, token, label):
        start = len(" ".join(self.tokens[0:token[0]])) + 1
        length = len(" ".join(self.tokens[token[0]:token[1]]))
        head = self.dict_head2span.get((token[0], token[1]), None)
        return start, start + length, label, head

    def tags_of_span_head(self, span):
        head_ix = self.dict_head2span.get(tuple(span))
        if head_ix is None:
            return [], []
        return self.deprel[head_ix], self.tags[head_ix]

    def _find_extra_spans(self, pred_overlaps):
        for ix, ent in enumerate(pred_overlaps):
            keys = ent[0].keys()
            for key in keys:
                if not any([x[key] for x in ent]):
                    self.pred_error_ix.add(ix)
                    self.extra_spans.append(self.pred[ix][key])

    def _find_missing_spans(self, gold_overlaps):
        tuplify = lambda x: [tuple(y) for y in x]
        flatten_overlap = lambda x: list(chain.from_iterable(chain.from_iterable([tuple(y for y in z if y) for z in x])))  # wildly ugly
        for ix, ent in enumerate(gold_overlaps):

            difference = set(tuplify(self.gold[ix])).difference(set(tuplify(flatten_overlap(ent))))
            if difference:
                self.gold_error_ix.add(ix)
            self.missing_spans.extend(difference)


    @property
    def missing_span_tags(self):
        deps = []
        tags = []
        for span in self.missing_spans:
            d, t = self.tags_of_span_head(span)
            if d:
                deps.append(d)
            if t:
                tags.append(t)
        return deps, tags

    @property
    def all_span_tags(self):
        deps = []
        tags = []
        for span in self.all_spans:
            d, t = self.tags_of_span_head(span)
            if d:
                deps.append(d)
            if t:
                tags.append(t)
        return deps, tags


# def link_entities(gold, pred, scores):


pred_path = Path("data/conll_logs/pred.json")
gold_path = Path("data/conll_logs/gold.json")

pred_sents = list(jsonlines.open(pred_path))
gold_sents = list(jsonlines.open(gold_path))

assert len(pred_sents) == len(gold_sents)

errors = {x: [] for x in ErrorType}
total_error_counts: tp.DefaultDict[ErrorType, int] = defaultdict(int)
ast, mst, asd, msd = [], [], [], []
for g, p in zip(gold_sents, pred_sents):
    if g["clusters"] and p["clusters"] and g["clusters"] != p["clusters"]:
        alignment = GoldPredAlignment(g["clusters"], p["clusters"], g["cased_words"], g["deprel"], g["head2span"], g["pos"])
        for key, value in alignment.error_counts.items():
            errors[key].append(alignment)
            total_error_counts[key] += value
        missing_dep, missing_tags = alignment.missing_span_tags
        all_dep, all_tags = alignment.all_span_tags
        ast.extend(all_tags)
        mst.extend(missing_tags)
        msd.extend(missing_dep)
        asd.extend(all_dep)

for key, value in errors.items():
    with Path(f"data/conll_logs/{key}.json").open("w") as io:
        io.writelines([x.error_line() + "\n" for x in value])


def pair_dicts(dict1, dict2):
    pair1 = []
    pair2 = []
    for key in dict1.keys():
        pair1.append(dict1[key])
        pair2.append(dict2.get(key, 1))
    return pair1, pair2


def normalize_dict(d):
    factor = 1.0 / sum(d.values())
    normalised_d = {k: v * factor for k, v in d.items()}
    return normalised_d


def pair_norm(dict1, dict2):
    normalized_d1 = normalize_dict(dict1)
    normalized_d2 = normalize_dict(dict2)
    res = {}
    for key in dict1.keys():
        res[key] = (normalized_d1[key] / normalized_d2.get(key, 1), dict1[key], dict2.get(key, 1))

    return res


normalize = lambda x: [float(i)/sum(x) for i in x]


with Path(f"data/tag_counts.json").open("w") as io:
    ast_subsample = random.sample(ast, len(mst))
    asd_subsample = random.sample(asd, len(asd))
    mst = dict(Counter(mst))
    ast = dict(Counter(ast))
    asts = dict(Counter(ast_subsample))
    asd = dict(Counter(asd))
    asds = dict(Counter(asd_subsample))
    msd = dict(Counter(msd))
    st_pairs = pair_dicts(mst, asts)
    sd_pairs = pair_dicts(msd, asds)
    st_dicts = pair_norm(mst, ast)
    sd_dicts = pair_norm(msd, asd)
    st_pairs_norm = [normalize(x) for x in st_pairs]
    sd_pairs_norm = [normalize(x) for x in sd_pairs]
    # chisquare_st = chisquare(st_pairs[0], st_pairs[1])
    # chisquare_sd = chisquare(sd_pairs[0], sd_pairs[1])
    print(total_error_counts)
    # print(chisquare_st)
    # print(chisquare_sd)
    print(ast, mst, asd, msd)
    print(st_pairs_norm, sd_pairs_norm)
    print(sorted(st_dicts.items(), key=lambda x: x[1][0]), sorted(sd_dicts.items(), key=lambda x: x[1][0]))

    io.writelines(json.dumps([ast, mst, asd, msd]))


# def align_entities(gold: tp.List[tp.List[int, int]], pred: tp.List[tp.List[int, int]]):
#     for ix, entity in gold:

