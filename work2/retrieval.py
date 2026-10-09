import json
import urllib.request

from .common import digest


class SearchR1Retriever:
    def __init__(self, url, timeout=120):
        self.url, self.timeout = url, timeout

    def search(self, query, top_k):
        body = json.dumps({"queries": [query], "topk": top_k, "return_scores": True}).encode()
        request = urllib.request.Request(self.url, data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
        # Local services must not accidentally use an outbound HTTP proxy.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=self.timeout) as response:
            payload = json.load(response)
        groups = payload.get("result")
        if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], list):
            raise ValueError("Expected Search-R1 response: {'result': [[document items]]}")
        documents = []
        for rank, item in enumerate(groups[0][:top_k], 1):
            if not isinstance(item, dict) or not isinstance(item.get("document"), dict):
                raise ValueError("Retriever must return scored {'document': ..., 'score': ...} items")
            raw = item["document"]
            contents = raw.get("contents")
            if not isinstance(contents, str) or not contents.strip():
                raise ValueError("Retriever document.contents is missing or empty")
            original_id = raw.get("id", raw.get("doc_id"))
            documents.append({"document_id": str(original_id) if original_id is not None else "sha256:" + digest(contents.encode()),
                              "id_basis": "upstream" if original_id is not None else "content_hash",
                              "title": contents.split("\n", 1)[0].strip().strip('"'),
                              "contents": contents, "score": item.get("score"), "rank": rank})
        return documents


def format_documents(documents):
    # Search-R1's document format; never include gold answers or supporting facts.
    parts = []
    for rank, document in enumerate(documents, 1):
        content = document["contents"]
        title = content.split("\n", 1)[0]
        text = content.split("\n", 1)[1] if "\n" in content else ""
        parts.append("Doc {}(Title: {}) {}\n".format(rank, title, text))
    return "".join(parts)
