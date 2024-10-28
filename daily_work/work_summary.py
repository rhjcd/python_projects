import sys
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QComboBox, QLabel, QFileDialog,
                             QCalendarWidget, QStackedWidget, QStyleFactory, QGroupBox, QButtonGroup)
from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QPalette
from datetime import datetime, timedelta
from jira import JIRA, JIRAError
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_USERNAME = os.getenv('GITHUB_USERNAME')
GITHUB_REPO = os.getenv('GITHUB_REPO')
GITHUB_API_BASE = os.getenv('GITHUB_API_BASE')
JIRA_TOKEN = os.getenv('JIRA_TOKEN')
JIRA_BASE_URL = os.getenv('JIRA_BASE_URL')
JIRA_PROJECT_ID = os.getenv('JIRA_PROJECT_ID')
JIRA_USERNAME = os.getenv('JIRA_USERNAME')

# Validation
if not all([GITHUB_TOKEN, GITHUB_USERNAME, GITHUB_REPO, GITHUB_API_BASE, 
            JIRA_TOKEN, JIRA_BASE_URL, JIRA_PROJECT_ID, JIRA_USERNAME]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

class WorkerThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        result = self.function(*self.args, **self.kwargs)
        self.update_signal.emit(result)

class WorkSummarizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker_threads = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Work Summarizer')
        self.setGeometry(100, 100, 800, 600)

        # Set the application style
        QApplication.setStyle(QStyleFactory.create('Fusion'))

        # Set color palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(palette)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title_label = QLabel("Work Summarizer")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 28, QFont.Bold))
        title_label.setStyleSheet("color: #2A82DA; margin: 20px 0;")
        main_layout.addWidget(title_label)

        # Period selection
        period_layout = QHBoxLayout()
        period_label = QLabel('Select Period:')
        period_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        period_layout.addWidget(period_label)
        self.period_combo = QComboBox()
        self.period_combo.addItems(['Daily', 'Weekly', 'Custom'])
        self.period_combo.setFixedWidth(120)
        self.period_combo.setStyleSheet("""
            QComboBox {
                border: 2px solid #FF6B6B;
                border-radius: 5px;
                padding: 5px;
                background: #FF6B6B;
                color: white;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        self.period_combo.currentIndexChanged.connect(self.on_period_change)
        period_layout.addWidget(self.period_combo)
        period_layout.addStretch()
        main_layout.addLayout(period_layout)

        # Date selection
        self.date_stack = QStackedWidget()
        self.setup_daily_widget()
        self.setup_weekly_widget()
        self.setup_custom_widget()
        main_layout.addWidget(self.date_stack)

        # Report type selection
        report_group = QGroupBox("Report Type")
        report_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)
        report_layout = QHBoxLayout(report_group)
        
        self.button_group = QButtonGroup(self)
        for text in ["GitHub", "Jira", "All"]:
            button = QPushButton(text)
            button.setCheckable(True)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 5px;
                    font-size: 14px;
                    min-width: 80px;
                    max-width: 80px;
                }
                QPushButton:checked {
                    background-color: #45a049;
                }
                QPushButton:hover {
                    background-color: #66BB6A;
                }
            """)
            self.button_group.addButton(button)
            report_layout.addWidget(button)
        
        self.button_group.buttons()[-1].setChecked(True)
        self.button_group.buttonClicked.connect(self.on_report_type_changed)
        
        main_layout.addWidget(report_group)

        # Action buttons
        button_layout = QHBoxLayout()
        for text in ['Generate Summary', 'Save Summary']:
            button = QPushButton(text)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #42A5F5;
                }
                QPushButton:pressed {
                    background-color: #1E88E5;
                }
            """)
            if text == 'Generate Summary':
                button.clicked.connect(self.generate_summary)
            else:
                button.clicked.connect(self.save_summary)
            button_layout.addWidget(button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Log output area
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 10px;
                font-family: Monaco, monospace;
                font-size: 14px;
            }
        """)
        self.log_output.setMinimumHeight(200)
        main_layout.addWidget(self.log_output)

    def on_report_type_changed(self, button):
        animation = QPropertyAnimation(button, b"geometry")
        animation.setDuration(100)
        animation.setStartValue(button.geometry().adjusted(2, 2, -2, -2))
        animation.setEndValue(button.geometry())
        animation.setEasingCurve(QEasingCurve.OutBounce)
        animation.start()

    def setup_daily_widget(self):
        daily_widget = QCalendarWidget()
        daily_widget.setGridVisible(True)
        daily_widget.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
        self.date_stack.addWidget(daily_widget)

    def setup_weekly_widget(self):
        weekly_widget = QCalendarWidget()
        weekly_widget.setGridVisible(True)
        weekly_widget.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
        weekly_widget.setSelectionMode(QCalendarWidget.NoSelection)
        self.date_stack.addWidget(weekly_widget)

    def setup_custom_widget(self):
        custom_widget = QWidget()
        layout = QHBoxLayout(custom_widget)
        self.start_calendar = QCalendarWidget()
        self.end_calendar = QCalendarWidget()
        self.start_calendar.setGridVisible(True)
        self.end_calendar.setGridVisible(True)
        self.start_calendar.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
        self.end_calendar.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
        layout.addWidget(self.start_calendar)
        layout.addWidget(self.end_calendar)
        self.date_stack.addWidget(custom_widget)

    def on_period_change(self, index):
        self.date_stack.setCurrentIndex(index)

    def generate_summary(self):
        period = self.period_combo.currentText()
        if period == 'Daily':
            start_date = end_date = datetime.now().strftime('%Y-%m-%d')
        elif period == 'Weekly':
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        else:  # Custom
            start_date = self.start_calendar.selectedDate().toString(Qt.ISODate)
            end_date = self.end_calendar.selectedDate().toString(Qt.ISODate)

        self.log_output.clear()
        self.log_output.append(f"Generating summary for period: {start_date} to {end_date}")

        selected_button = self.button_group.checkedButton()
        if selected_button.text() == "GitHub" or selected_button.text() == "All":
            worker = WorkerThread(self.get_github_activity, start_date, end_date)
            worker.update_signal.connect(self.update_log)
            self.worker_threads.append(worker)
            worker.start()

        if selected_button.text() == "Jira" or selected_button.text() == "All":
            worker = WorkerThread(self.get_jira_activity, start_date, end_date)
            worker.update_signal.connect(self.update_log)
            self.worker_threads.append(worker)
            worker.start()

    def update_log(self, message):
        self.log_output.append(message)

    def closeEvent(self, event):
        for thread in self.worker_threads:
            thread.quit()
            thread.wait()
        event.accept()

    def get_github_activity(self, start_date, end_date):
        debug_info = f"Debug Info:\nGITHUB_TOKEN: {GITHUB_TOKEN[:5]}...\n"
        debug_info += f"GITHUB_USERNAME: {GITHUB_USERNAME}\n"
        debug_info += f"GITHUB_REPO: {GITHUB_REPO}\n"
        debug_info += f"GITHUB_API_BASE: {GITHUB_API_BASE}\n"
        debug_info += f"Date Range: {start_date} to {end_date}\n\n"
    
        try:
            headers = {
                'Authorization': f'token {GITHUB_TOKEN}',
                'Accept': 'application/vnd.github.v3+json'
            }
    
            # Fetch pull requests created by the specific user
            pr_url = f"{GITHUB_API_BASE}/search/issues"
            pr_params = {
                'q': f'repo:{GITHUB_REPO} is:pr author:{GITHUB_USERNAME} created:{start_date}..{end_date}',
                'sort': 'created',
                'order': 'desc'
            }
            debug_info += f"Requesting URL: {pr_url}\n"
            debug_info += f"With parameters: {pr_params}\n"
    
            pr_response = requests.get(pr_url, headers=headers, params=pr_params)
            pr_response.raise_for_status()
            search_results = pr_response.json()
    
            debug_info += f"Total PRs fetched: {search_results['total_count']}\n"
    
            categorized_prs = {}
            for item in search_results['items']:
                pr_number = item['number']
                pr_title = item['title']
                pr_state = item['state']
                pr_created_date = item['created_at'][:10]
    
                debug_info += f"\nChecking PR #{pr_number}:\n"
                debug_info += f"  Created at: {pr_created_date}\n"
                debug_info += f"  Current state: {pr_state}\n"
    
                # Fetch additional PR details
                pr_detail_url = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/pulls/{pr_number}"
                pr_detail_response = requests.get(pr_detail_url, headers=headers)
                pr_detail_response.raise_for_status()
                pr_detail = pr_detail_response.json()
    
                if pr_state == 'closed' and pr_detail.get('merged_at'):
                    status = 'Merged'
                elif pr_state == 'closed':
                    status = 'Closed'
                else:
                    status = 'Open'
    
                # Fetch files changed in the PR
                files_url = f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/pulls/{pr_number}/files"
                files_response = requests.get(files_url, headers=headers)
                files_response.raise_for_status()
                files = files_response.json()
                folder = files[0]['filename'].split('/')[0] if files else 'Unknown'
    
                if folder not in categorized_prs:
                    categorized_prs[folder] = []
                categorized_prs[folder].append(f"PR #{pr_number}: {pr_title} ({status})")
    
                debug_info += f"  Included in summary (Status: {status}, Folder: {folder})\n"
    
            summary = "GitHub Activity:\n"
            for folder, prs in categorized_prs.items():
                summary += f"\n{folder}:\n"
                for pr in prs:
                    summary += f"  - {pr}\n"
    
            if not categorized_prs:
                summary += "No GitHub activity found for the specified period.\n"
    
            return debug_info + "\n" + summary
    
        except requests.RequestException as e:
            return debug_info + f"\nError fetching GitHub data: {str(e)}"
    
    def get_jira_activity(self, start_date, end_date):
        try:
            jira = JIRA(server=JIRA_BASE_URL, token_auth=JIRA_TOKEN)
            
            jql = f'project = {JIRA_PROJECT_ID} AND assignee = {JIRA_USERNAME} AND updated >= "{start_date}" AND updated <= "{end_date}" ORDER BY updated DESC'
            issues = jira.search_issues(jql, expand='changelog')
            
            if not issues:
                return "No Jira activity found for the specified period."

            summary = "Jira Activity:\n"
            total_time_spent = 0
            for issue in issues:
                time_spent = self.get_time_spent(issue, start_date, end_date)
                total_time_spent += time_spent
                summary += f"- {issue.key}: {issue.fields.summary} (Status: {issue.fields.status.name})\n"
                summary += f"  Time spent: {time_spent} minutes\n"
             
            summary += f"\nTotal time spent: {total_time_spent} minutes ({total_time_spent/60:.2f} hours)\n"
            return summary

        except JIRAError as e:
            return f"Error fetching Jira data: {str(e)}"
    
    def get_time_spent(self, issue, start_date, end_date):
        time_spent = 0
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)

        for history in issue.changelog.histories:
            for item in history.items:
                if item.field == 'timespent':
                    change_time = datetime.strptime(history.created[:10], '%Y-%m-%d')
                    if start_datetime <= change_time < end_datetime:
                        time_spent += int(item.to or 0) - int(item.from_ or 0)

        return time_spent // 60  # Convert seconds to minutes

    def save_summary(self):
        summary = self.log_output.toPlainText()
        if not summary:
            self.log_output.append("No summary to save. Generate a summary first.")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Save Summary", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            with open(file_name, 'w') as file:
                file.write(summary)
            self.log_output.append(f"Summary saved to {file_name}")

def main():
    app = QApplication(sys.argv)
    ex = WorkSummarizer()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
