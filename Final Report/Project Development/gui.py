import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QComboBox, QProgressBar, QHBoxLayout, QSpacerItem,
    QSizePolicy, QDialog, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QMovie
from src.docx_reader import extract_text_from_docx
from src.csv_reader import extract_text_from_csv
from src.pdf_reader import extract_text_from_pdf
from src.text_preprocessing import preprocess_text
from src.lda_theme_extraction import perform_lda
from src.ner_module import extract_entities
from src.nmf_module import perform_nmf
from src.sentiment_analysis import analyze_sentiment
from src.result_export import export_results_to_csv, export_results_to_docx
import nltk
import time

from PyQt5.QtCore import QThread, pyqtSignal


nltk.download('punkt')
nltk.download('vader_lexicon')


class Worker(QThread):
    progress_updated = pyqtSignal(int)
    task_completed = pyqtSignal(list)
    task_failed = pyqtSignal(str)

    def __init__(self, text, chunk_size):
        super().__init__()
        self.text = text
        self.chunk_size = chunk_size

    def run(self):
        try:
            processed_text = []
            total_length = len(self.text)
            for i in range(10):
                chunk_start = i * self.chunk_size
                chunk_end = min(chunk_start + self.chunk_size, total_length)
                chunk = self.text[chunk_start:chunk_end]

                # Simulate processing (replace this with actual preprocessing logic)
                processed_chunk = preprocess_text(chunk)  # Call your preprocessing function
                processed_text.extend(processed_chunk)

                # Simulate processing time
                time.sleep(0.5)  # Delay to mimic processing time


                progress = int(((i + 1) / 10) * 100)
                self.progress_updated.emit(progress)

            self.task_completed.emit(processed_text)
        except Exception as e:
            self.task_failed.emit(str(e))



class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qualitative Data Analysis")
        self.setGeometry(100, 100, 800, 600)
        self.set_background()
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)

        self.title_label = QLabel("", self)
        self.title_label.setStyleSheet("font-size: 32px; font-weight: bold; color: darkblue; font-family: 'Comic Sans MS';")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label, alignment=Qt.AlignTop)

        self.gif_label = QLabel(self)
        self.gif = QMovie("images/data_scene.gif")
        self.gif_label.setMovie(self.gif)
        self.gif.setScaledSize(self.size())
        self.gif_label.setScaledContents(True)
        self.gif_label.setFixedSize(800, 600)
        self.gif.start()
        self.layout.addWidget(self.gif_label)

        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.title_text = "Qualitative Data Analysis"
        self.current_index = 0
        self.typing_timer = QTimer(self)
        self.typing_timer.timeout.connect(self.type_title)
        self.typing_timer.start(100)

        self.gif_duration = self.calculate_gif_duration()
        self.stop_timer = QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self.gif_finished)
        self.stop_timer.start(self.gif_duration)

        self.start_button = QPushButton("Start", self)
        self.start_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: black;
                font-size: 20px;
                padding: 10px;
                border: none;
            }
            QPushButton:hover {
                text-decoration: underline;
            }
        """)
        arrow_icon = QIcon("images/next.png")
        self.start_button.setIcon(arrow_icon)
        self.start_button.setIconSize(QSize(20, 20))
        self.start_button.clicked.connect(self.open_main_window)
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        self.gif.finished.connect(self.gif_finished)
    
    def calculate_gif_duration(self):
        return 4500

    def type_title(self):
        if self.current_index < len(self.title_text):
            self.title_label.setText(self.title_label.text() + self.title_text[self.current_index])
            self.current_index += 1
        else:
            self.typing_timer.stop()

    def gif_finished(self):
        self.gif.stop()
        self.start_button.setEnabled(True)

    def open_main_window(self):
        self.main_window = TextAnalysisApp()
        self.main_window.show()
        self.close()

    def set_background(self):
        self.setStyleSheet("background-color: white;")


class TextAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Analysis Tool")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("images/data-analysis.png"))
        self.set_background()

        self.file_path = None
        self.file_type = None
        self.preprocessed_text = None
        self.results = {}
        self.preprocessed_once = False

        self.create_widgets()
        self.show()

    def set_background(self):
        self.setStyleSheet("background-color: white;")

    def create_widgets(self):
        layout = QVBoxLayout()

        title_label = QLabel("Automated Text Analysis Tool", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #000;")
        layout.addWidget(title_label)

        upload_layout = QHBoxLayout()
        self.upload_button = self.create_button("Upload File", self.upload_file)
        upload_layout.addWidget(self.upload_button)

        self.file_label = QLabel("No file uploaded", self)
        self.file_label.setStyleSheet("color: #000;")
        upload_layout.addWidget(self.file_label)

        layout.addLayout(upload_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                text-align: center;
                background-color: #555;
                border-radius: 10px;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.preprocess_button = self.create_button("Process Text", self.process_text)
        layout.addWidget(self.preprocess_button)

        theme_layout = QHBoxLayout()
        self.theme_method = QComboBox(self)
        self.theme_method.addItems(["LDA", "NER", "NMF"])
        theme_layout.addWidget(QLabel("Select Theming Method:", self))

        theme_layout.addWidget(self.theme_method)
        self.theme_button = self.create_button("Analyze Theme", self.theme_analysis)
        theme_layout.addWidget(self.theme_button)

        layout.addLayout(theme_layout)

        self.analyze_button = self.create_button("Sentiment Analysis", self.sentiment_analysis)
        layout.addWidget(self.analyze_button)

        download_layout = QHBoxLayout()
        self.download_format = QComboBox(self)
        self.download_format.addItems(["CSV", "DOCX"])
        download_layout.addWidget(QLabel("Select Download Format:", self))

        download_layout.addWidget(self.download_format)
        self.download_button = self.create_button("Download Results", self.download_results)
        download_layout.addWidget(self.download_button)

        # Add Preview Results Button
        self.preview_button = self.create_button("Preview Results", self.preview_results)
        layout.addWidget(self.preview_button)

        layout.addLayout(download_layout)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def create_button(self, text, slot):
        button = QPushButton(text, self)
        button.clicked.connect(slot)
        button.setStyleSheet("""
            QPushButton {
                background-color: rgba(74, 122, 188, 0.8);
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: rgba(74, 122, 188, 1);
            }
        """)
        button.setFixedHeight(40)
        return button

    def upload_file(self):
        new_file_path, _ = QFileDialog.getOpenFileName(self, "Upload File", "", "Text Files (*.docx *.csv *.pdf)")
        if new_file_path:
            self.file_path = new_file_path
            self.file_label.setText(self.file_path)
            self.file_type = self.file_path.split('.')[-1]
            self.preprocessed_text = None
            self.results.clear()
            self.preprocessed_once = False

    def process_text(self):
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "No file uploaded!")
            return

        if self.preprocessed_once:
            QMessageBox.warning(self, "Warning", "Text has already been preprocessed.")
            return

        try:
            text = self.load_text()
            total_length = len(text)
            chunk_size = total_length // 10

            self.progress_bar.setValue(0)

            # Create a Worker thread
            self.worker = Worker(text, chunk_size)

            # Connect signals from the worker to update the GUI
            self.worker.progress_updated.connect(self.progress_bar.setValue)
            self.worker.task_completed.connect(self.on_processing_complete)
            self.worker.task_failed.connect(self.on_processing_failed)

            # Start the worker thread
            self.worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_processing_complete(self, processed_text):
        self.preprocessed_text = processed_text
        self.results['preprocessed_text'] = processed_text
        self.preprocessed_once = True
        QMessageBox.information(self, "Success", "Text processing complete.")

    def on_processing_failed(self, error_message):
        QMessageBox.critical(self, "Error", f"Text processing failed: {error_message}")

       

    def theme_analysis(self):
        if 'preprocessed_text' not in self.results:
            QMessageBox.warning(self, "Warning", "Preprocessed text is required!")
            return

        method = self.theme_method.currentText()
        try:
            if method == "LDA":
                topics = perform_lda(self.preprocessed_text)
                self.results['lda_topics'] = topics
            elif method == "NER":
                if isinstance(self.preprocessed_text, list):
                    entities = extract_entities(" ".join(self.preprocessed_text))
                else:
                    entities = extract_entities(self.preprocessed_text)
                self.results['ner_entities'] = entities
            elif method == "NMF":
                topics = perform_nmf(self.preprocessed_text)
                self.results['nmf_topics'] = topics
            
            QMessageBox.information(self, "Success", f"{method} analysis complete.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def sentiment_analysis(self):
        if 'preprocessed_text' not in self.results:
            QMessageBox.warning(self, "Warning", "Preprocessed text is required!")
            return

        try:
            if isinstance(self.preprocessed_text, list):
                self.preprocessed_text = " ".join(self.preprocessed_text)

            sentiments = analyze_sentiment(self.preprocessed_text)
            self.results['sentiment_analysis'] = sentiments
            QMessageBox.information(self, "Success", "Sentiment analysis complete.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def download_results(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to download!")
            return

        format_type = self.download_format.currentText()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # Prompt the user to select the save location and file name
        file_filter = "CSV Files (*.csv)" if format_type == "CSV" else "DOCX Files (*.docx)"
        save_path, _ = QFileDialog.getSaveFileName(self, 
                                                    "Save Results", 
                                                    "",  # Default filename suggestion can be added here
                                                    file_filter,
                                                    options=options)
        if not save_path:  # User canceled the dialog
            return

        # Ensure correct file extension is added if missing
        if format_type == "CSV" and not save_path.endswith(".csv"):
            save_path += ".csv"
        elif format_type == "DOCX" and not save_path.endswith(".docx"):
            save_path += ".docx"

        try:
            if format_type == "CSV":
                export_results_to_csv(self.results, save_path)
                QMessageBox.information(self, "Success", f"Results exported to {save_path}")
            elif format_type == "DOCX":
                export_results_to_docx(self.results, save_path)
                QMessageBox.information(self, "Success", f"Results exported to {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {e}")


    def load_text(self):
        if self.file_type == 'docx':
            return extract_text_from_docx(self.file_path)
        elif self.file_type == 'csv':
            return extract_text_from_csv(self.file_path)
        elif self.file_type == 'pdf':
            return extract_text_from_pdf(self.file_path)
        return ""

    def preview_results(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to preview!")
            return

        result_text = "\n\n".join([f"{key}:\n{value}" for key, value in self.results.items()])

        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Preview Results")
        preview_dialog.setGeometry(150, 150, 600, 400)

        layout = QVBoxLayout()
        result_display = QTextEdit(preview_dialog)
        result_display.setReadOnly(True)
        result_display.setText(result_text)
        
        layout.addWidget(result_display)
        preview_dialog.setLayout(layout)
        preview_dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()

    splash.gif.finished.connect(lambda: splash.open_main_window())

    sys.exit(app.exec_())