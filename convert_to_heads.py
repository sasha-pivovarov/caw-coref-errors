from collections import defaultdict
import logging
import os
from typing import Tuple
from functools import lru_cache

import jsonlines


DATA_DIR = "data"
FILENAME = "english_{}{}.jsonlines"
LOGGING_LEVEL = logging.WARNING  # DEBUG to output all duplicate spans
SPLITS = ("development", "test", "train")

class DynamicDepth():
    """Implements a cache + dynamic programming to find the relative depth of every word in a subphrase given the head word for every word.
    """
    def get_parse_depths(self, heads, start, end):
        """Return the relative depth for every word

        Args:
            heads (list): List where each entry is the index of that entry's head word in the dependency parse
            start (int): starting index of the heads for the subphrase
            end (int): ending index of the heads for the subphrase

        Returns:
            list: Relative depth in the dependency parse for every word
        """
        self.heads = heads[start:end]
        self.relative_heads = [h - start if h else -100 for h in self.heads] # -100 to deal with 'none' headwords

        depths = [self._get_depth_recursive(h) for h in range(len(self.relative_heads))]

        return depths

    @lru_cache(maxsize=None)
    def _get_depth_recursive(self, index):
        """Recursively get the depths of every index using a cache and recursion

        Args:
            index (int): Index of the word for which to calculate the relative depth

        Returns:
            int: Relative depth of the word at the index
        """
        # if the head for the current index is outside the scope, this index is a relative root
        if self.relative_heads[index] >= len(self.relative_heads) or self.relative_heads[index] < 0:
            return 0
        return self._get_depth_recursive(self.relative_heads[index]) + 1
    
def get_head(mention: Tuple[int, int], doc: dict) -> int:
    """Returns the span's head, which is defined as the only word within the
    span whose head is outside of the span or None. In case there are no or
    several such words, the rightmost word is returned

    Args:
        mention (Tuple[int, int]): start and end (exclusive) of a span
        doc (dict): the document data

    Returns:
        int: word id of the spans' head
    """
    head_candidates = set()
    start, end = mention

    # use head information to extract parse depth
    dynamicDepth = DynamicDepth()
    depth = dynamicDepth.get_parse_depths(doc['head'], start, end)
    depth_limit = 2

    # return first 'CC' token above depth limit, if exists
    cc_indexes = [i for i in range(end - start) if doc['pos'][i+start] == 'CC' and depth[i] < depth_limit]
    if cc_indexes:
        return cc_indexes[0] + start

    # else return syntactic head
    for i in range(start, end):
        ith_head = doc["head"][i]
        if ith_head is None or not (start <= ith_head < end):
            head_candidates.add(i)
    if len(head_candidates) == 1:
        return head_candidates.pop()
    return end - 1


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    path = os.path.join(DATA_DIR, FILENAME)
    for split in SPLITS:
        with jsonlines.open(path.format(split, ""), mode="r") as inf:
            with jsonlines.open(path.format(split, "_head"), mode="w") as outf:
                deleted_spans = 0
                deleted_clusters = 0
                total_spans = 0
                total_clusters = 0

                for doc in inf:

                    total_spans += sum(len(cluster) for cluster in doc["clusters"])
                    total_clusters += len(doc["clusters"])

                    head_clusters = [
                        [get_head(mention, doc) for mention in cluster]
                        for cluster in doc["clusters"]
                    ]

                    # check for duplicates
                    head2spans = defaultdict(list)
                    for cluster, head_cluster in zip(doc["clusters"], head_clusters):
                        for span, span_head in zip(cluster, head_cluster):
                            head2spans[span_head].append((span, head_cluster))

                    doc["head2span"] = []

                    for head, spans in head2spans.items():
                        spans.sort(key=lambda x: x[0][1] - x[0][0])  # shortest spans first
                        doc["head2span"].append((head, *spans[0][0]))

                        if len(spans) > 1:
                            logging.debug(f'{split} {doc["document_id"]} {doc["cased_words"][head]}')
                            for span, cluster in spans:
                                logging.debug(f'{id(cluster)} {" ".join(doc["cased_words"][slice(*span)])}')
                            logging.debug("=====")

                            for _, cluster in spans[1:]:
                                cluster.remove(head)
                                deleted_spans += 1

                    filtered_head_clusters = [cluster for cluster in head_clusters if len(cluster) > 1]
                    deleted_clusters += len(head_clusters) - len(filtered_head_clusters)
                    doc["word_clusters"] = filtered_head_clusters
                    doc["span_clusters"] = doc["clusters"]
                    del doc["clusters"]

                    outf.write(doc)

                print(f"Deleted in {split}:"
                      f"\n\t{deleted_spans}/{total_spans} ({deleted_spans/total_spans:.2%}) spans"
                      f"\n\t{deleted_clusters}/{total_clusters} ({deleted_clusters/total_clusters:.2%}) clusters"
                      f"\n")
