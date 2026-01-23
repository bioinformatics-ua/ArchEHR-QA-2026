from pathlib import Path
import xml.etree.ElementTree as ET

# TODO I would like this to be complete dataloader - the paths can be coded directly... 
class ArchEHRDataLoader:
    def __init__(self, xml_path: Path):
        self.xml_path = xml_path

    def load(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        return [
            {
                "case_id": case.get("id"),
                "clinician_question": text,
            }
            for case in root.findall("case")
            if (text := case.findtext("clinician_question", "").strip())
        ]
