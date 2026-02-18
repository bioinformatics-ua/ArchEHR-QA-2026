import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Phrase:
    """Represents a phrase within a patient question."""

    id: str
    start_char_index: int
    text: str


@dataclass
class Sentence:
    """Represents a sentence from the clinical note excerpt."""

    id: str
    paragraph_id: str
    start_char_index: int
    length: int
    text: str


@dataclass
class Case:
    """Represents a complete ArchEHR-QA case."""

    case_id: str
    clinical_specialty: str
    patient_narrative: str
    patient_question_phrases: list[Phrase]
    clinician_question: str
    note_excerpt: str
    sentences: list[Sentence]


class ArchEHRDataLoader:
    """Loads and parses ArchEHR-QA XML files into structured Case objects."""

    def __init__(self, xml_path: Path):
        self.xml_path = xml_path

    def load(self) -> list[Case]:
        """
        Load all cases from the XML file.

        Returns:
            List of Case objects containing all parsed data.
        """
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        cases = []
        for case_elem in root.findall("case"):
            case = self._parse_case(case_elem)
            if case:
                cases.append(case)

        return cases

    def _parse_case(self, case_elem: ET.Element) -> Case | None:
        """Parse a single case element into a Case object."""
        case_id = case_elem.get("id")
        if not case_id:
            return None

        # Extract basic text fields
        clinical_specialty = case_elem.findtext("clinical_specialty", "").strip()
        patient_narrative = case_elem.findtext("patient_narrative", "").strip()
        clinician_question = case_elem.findtext("clinician_question", "").strip()
        note_excerpt = case_elem.findtext("note_excerpt", "").strip()

        # Parse patient question phrases
        phrases = []
        patient_question = case_elem.find("patient_question")
        if patient_question is not None:
            for phrase_elem in patient_question.findall("phrase"):
                phrase = self._parse_phrase(phrase_elem)
                if phrase:
                    phrases.append(phrase)

        # Parse note excerpt sentences
        sentences = []
        note_excerpt_sentences = case_elem.find("note_excerpt_sentences")
        if note_excerpt_sentences is not None:
            for sentence_elem in note_excerpt_sentences.findall("sentence"):
                sentence = self._parse_sentence(sentence_elem)
                if sentence:
                    sentences.append(sentence)

        return Case(
            case_id=case_id,
            clinical_specialty=clinical_specialty,
            patient_narrative=patient_narrative,
            patient_question_phrases=phrases,
            clinician_question=clinician_question,
            note_excerpt=note_excerpt,
            sentences=sentences,
        )

    def _parse_phrase(self, phrase_elem: ET.Element) -> Phrase | None:
        """Parse a phrase element from patient_question."""
        phrase_id = phrase_elem.get("id")
        start_char = phrase_elem.get("start_char_index")
        text = phrase_elem.text.strip() if phrase_elem.text else ""

        if phrase_id is None or start_char is None:
            return None

        try:
            start_char_index = int(start_char)
        except ValueError:
            return None

        return Phrase(
            id=phrase_id,
            start_char_index=start_char_index,
            text=text,
        )

    def _parse_sentence(self, sentence_elem: ET.Element) -> Sentence | None:
        """Parse a sentence element from note_excerpt_sentences."""
        sentence_id = sentence_elem.get("id")
        paragraph_id = sentence_elem.get("paragraph_id")
        start_char = sentence_elem.get("start_char_index")
        length = sentence_elem.get("length")
        text = sentence_elem.text.strip() if sentence_elem.text else ""

        if not all([sentence_id, paragraph_id, start_char, length]):
            return None

        try:
            start_char_index = int(start_char or 0)
            length_int = int(length or 0)
        except ValueError:
            return None

        return Sentence(
            id=sentence_id or "",
            paragraph_id=paragraph_id or "",
            start_char_index=start_char_index,
            length=length_int,
            text=text,
        )
