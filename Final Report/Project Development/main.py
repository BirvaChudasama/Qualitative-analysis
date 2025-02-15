from src.docx_reader import extract_text_from_docx
from src.csv_reader import extract_text_from_csv
from src.pdf_reader import extract_text_from_pdf
from src.text_preprocessing import preprocess_text
from src.lda_theme_extraction import perform_lda
from src.ner_module import extract_entities
from src.nmf_module import perform_nmf
from src.sentiment_analysis import analyze_sentiment
from src.result_export import export_results_to_csv, export_results_to_docx


def process_file(file_path, file_type):
    """
    Process the input file (docx, csv, pdf), perform preprocessing,
    extract themes with LDA and NMF, perform Named Entity Recognition (NER),
    and perform Sentiment Analysis. Return all results in a dictionary.
    """
    # Step 1: Read the file based on its type
    if file_type == 'docx':
        combined_text = extract_text_from_docx(file_path)
    elif file_type == 'csv':
        combined_text = extract_text_from_csv(file_path, 'text')
    elif file_type == 'pdf':
        combined_text = extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .docx, .csv, or .pdf file.")

    # Step 2: Preprocess Text (returns a list of tokens)
    preprocessed_text = preprocess_text(combined_text)

    # Step 3: Theme Extraction with LDA
    lda_model, lda_vectorizer, lda_topics = perform_lda([preprocessed_text], num_topics=5)

    # Step 4: Named Entity Recognition (NER)
    entities = extract_entities(" ".join(preprocessed_text))
    results['entities'] = entities
    
    # Step 5: Theme Extraction with NMF
    nmf_model, nmf_vectorizer, nmf_topics = perform_nmf([preprocessed_text], num_topics=5)

    # Step 6: Sentiment Analysis
    sentiment = analyze_sentiment(' '.join(preprocessed_text))  # Sentiment analysis needs a string

    # Step 7: Store all results (Preprocessed Text, LDA, NMF, NER, and Sentiment Analysis)
    results = {
        'preprocessed_text': ' '.join(preprocessed_text),  # Save preprocessed text as a single string
        'lda_topics': lda_topics,                         # LDA topics
        'nmf_topics': nmf_topics,                         # NMF topics
        'entities': entities,                             # Named Entities (NER)
        'sentiment': sentiment,                           # Sentiment analysis results
    }

    return results


def export_results(results):
    """
    Export the analysis results (LDA, NMF, NER, Sentiment, Preprocessed Text) to CSV and DOCX formats.
    """
    export_results_to_csv(results, 'data/output/results.csv')
    export_results_to_docx(results, 'data/output/results.docx')


if __name__ == "__main__":
    # Example usage (you can test this with actual file paths and types):
    # file_path = "path/to/your/file.docx"
    # file_type = "docx"  # Change this to 'csv' or 'pdf' if needed

    # Run the process and export results
    # results = process_file(file_path, file_type)
    # export_results(results)

    pass  # Placeholder to prevent execution when imported
