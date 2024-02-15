from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
import streamlit as st

def anonymize_text(text):
    analyzer = AnalyzerEngine()
    entities = analyzer.get_supported_entities() #get default entities
    # Call analyzer to get results
    results = analyzer.analyze(text=text,
                            entities=entities,
                            language='en',
                            score_threshold=0.35)
    anonymizer = AnonymizerEngine()
    anonymized_text = anonymizer.anonymize(text=text,analyzer_results=results)
    return anonymized_text.text
