import re
from typing import Dict, Optional
from difflib import SequenceMatcher

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def find_closest_strings(
    source_string, target_strings, k=3, model_name="all-MiniLM-L6-v2"
):
    model = None

    source_embedding = model.encode([source_string])
    target_embeddings = model.encode(target_strings)

    similarities = cosine_similarity(source_embedding, target_embeddings)[0]

    top_k_indices = np.argsort(similarities)[-k:][::-1]

    return [(target_strings[i], similarities[i]) for i in top_k_indices]


def extract_labeled_sections(text: str) -> Optional[Dict[str, str]]:
    pattern = r'\*\*(.*?):\*\*\s*"""\s*(.*?)\s*"""'

    matches = re.finditer(pattern, text, re.DOTALL)

    sections = {}
    for match in matches:
        label = match.group(1).strip()
        content = match.group(2).strip()
        sections[label] = content

    return sections if sections else None


def preprocess_text(text):
    text = re.sub(r"[^\w\s\']", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_longest_common_substring(str1, str2):
    seqMatch = SequenceMatcher(None, str1, str2)
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))
    return match.size


def find_best_matching_line(evidence, text_lines):
    if not evidence.strip():
        return -1, 0

    processed_evidence = preprocess_text(evidence)
    evidence_words = set(processed_evidence.split())

    best_match_score = 0
    best_line_num = -1

    for i, line in enumerate(text_lines):
        processed_line = preprocess_text(line)

        if not processed_line:
            continue

        lcs_score = get_longest_common_substring(processed_evidence, processed_line)

        line_words = set(processed_line.split())
        word_overlap = (
            len(evidence_words.intersection(line_words)) / len(evidence_words)
            if evidence_words
            else 0
        )

        sequence_ratio = SequenceMatcher(
            None, processed_evidence, processed_line
        ).ratio()

        combined_score = lcs_score * 0.4 + word_overlap * 0.3 + sequence_ratio * 0.3

        if combined_score > best_match_score:
            best_match_score = combined_score
            best_line_num = i

    return best_line_num, best_match_score


def split_into_sentences(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"([.!?])\s+([A-Z])", r"\1\n\2", text)

    text = re.sub(r";\s+", ";\n", text)

    text = re.sub(r"\n\s*\n", "\n", text)

    text = re.sub(r"(?m)^\s*[-•*]\s+", "\n", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "\n", text)

    lines = [line.strip() for line in text.split("\n")]

    MIN_LINE_LENGTH = 20
    result = []
    current_line = ""

    for line in lines:
        if not line.strip():
            continue

        if len(current_line) < MIN_LINE_LENGTH and current_line:
            current_line += " " + line
        else:
            if current_line:
                result.append(current_line)
            current_line = line

    if current_line:
        result.append(current_line)

    return result


def find_evidence_line_number(evidence, human_caption):
    lines = split_into_sentences(human_caption)

    line_num, score = find_best_matching_line(evidence, lines)

    threshold = 0.3
    return line_num if score >= threshold else None


def test_parser(text, parsed_results):
    line_count = len(re.findall(r"Line\s+\d+:", text))

    success = len(parsed_results) == line_count
    assert success

    print(f"Original line count: {line_count}")
    print(f"Parsed results count: {len(parsed_results)}")
    print(f"Test {'Passed' if success else 'Failed'}")

    if not success:
        for i, (
            line_num,
            line_txt,
            evidence,
            type_val,
            reasoning,
            verdict,
        ) in enumerate(parsed_results):
            print(f"\nEntry {i+1}:")
            print(f"  Line {line_num}: {line_txt}")
            print(f"  Type: {type_val}")
            print(f"  Evidence: {evidence}")
            print(f"  Reasoning: {reasoning}")
            print(f"  Verdict: {verdict}")

    return success, parsed_results


def parse_structured_text(text):
    text = text.strip().strip("```")

    line_blocks = re.split(r"(?=\nLine \d+:)", text)

    line_blocks = [block.strip() for block in line_blocks if block.strip()]

    results = []

    for block in line_blocks:
        line_match = re.match(r"Line\s+(\d+):\s*(.*?)(?=\n\s+-)", block, re.DOTALL)
        if not line_match:
            continue

        line_num = int(line_match.group(1))
        line_txt = line_match.group(2).strip()

        type_match = re.search(r"-\s*Type:\s*(.*?)(?=\n\s+-|\n\s*$)", block, re.DOTALL)
        type_val = type_match.group(1).strip() if type_match else ""

        evidence_match = re.search(
            r"-\s*Evidence:\s*(.*?)(?=\n\s+-|\n\s*$)", block, re.DOTALL
        )
        evidence_val = evidence_match.group(1).strip() if evidence_match else ""

        reasoning_match = re.search(
            r"-\s*Reasoning:\s*(.*?)(?=\n\s+-|\n\s*$)", block, re.DOTALL
        )
        reasoning_val = reasoning_match.group(1).strip() if reasoning_match else ""

        verdict_match = re.search(
            r"-\s*Verdict:\s*(.*?)(?=\n\s*$|\n\s*\n|$)", block, re.DOTALL
        )
        verdict_val = verdict_match.group(1).strip() if verdict_match else ""

        results.append(
            (line_num, line_txt, evidence_val, type_val, reasoning_val, verdict_val)
        )

    results.sort(key=lambda x: x[0])
    test_parser(text, results)
    return results


def filter_content_line(line, min_words=3, common_phrase_threshold=0.7):
    common_phrases = [
        r"here(?:'s| is) a(?:n)? (?:detailed |brief |quick |)?description of the video",
        r"overall (?:impression|summary|description|analysis)",
        r"visual details",
        r"overall effect",
        r"in summary",
        r"to summarize",
        r"\*\*.*?\*\*\s*:?$",
        r"here(?:'s| is) a(?:n)? (?:detailed |brief |quick |)?description of the video based on the images provided",
        r"here's a detailed description of the video based on the images provided",
        r"here is a detailed description of the images you provided",
        r"here's a detailed description of the video",
        r"here's a detailed description of the video, capturing its key elements",
        r"here's a detailed description of the video, based on the images provided",
    ]

    if not line.strip():
        return False

    clean_line = re.sub(r"\*+", "", line)
    words = [w for w in clean_line.split() if w.strip()]
    word_count = len(words)

    if word_count < min_words:
        return False

    line_lower = line.lower()

    line_lower = re.sub(r"[:;]", "", line_lower)

    for pattern_str in common_phrases:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        match = pattern.search(line_lower)

        if match:
            match_length = match.end() - match.start()
            content_length = len(line.strip())

            if match_length / content_length > common_phrase_threshold:
                return False
    return True


def compute_words_in_text(text: str):
    return len(text.split())
