import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from tkinter import ttk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, word_tokenize, ne_chunk
from nltk.tree import Tree
from string import punctuation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import dates as mdates
from wordcloud import WordCloud
import spacy

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

nlp = spacy.load("en_core_web_sm")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'DejaVu Sans'

class DataAnalyzerApp:
   
    def __init__(self, root):
        self.root = root
        self.root.title("Natural Language Processing (NLP)")
        self.root.geometry("1400x1000")
        self.root.configure(bg="#F0F0F0")

        # Default rows per page options
        self.rows_per_page_options = [5, 10, 25, 50, 100, 500]
        self.rows_per_page = tk.IntVar(value=10)
        self.current_page = 0  #

        # Notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        # Dataset View Tab
        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text="Dataset View")

        # Analysis View Tab
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Text Pre-Processing")
        self.notebook.tab(1, state="disabled")

        # Visualization Tab
        self.visualization_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_tab, text="Visualization")
        self.notebook.tab(2, state="disabled")

        # Vectorization Tab
        self.vectorization_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.vectorization_tab, text="Text Representation")
        self.notebook.tab(3, state="disabled")

        # Model Application Tab
        self.model_application_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_application_tab, text="Model Application")
        self.notebook.tab(4, state="disabled") 

        # Setup Tabs
        self.setup_dataset_view()
        self.setup_analysis_view()
        self.setup_visualization_view()
        self.setup_vectorization_view()
        self.setup_model_application_view()

        # Data placeholders
        self.dataframe = None
        self.processed_data = None
        self.vectorized_data = None

    def setup_dataset_view(self):
        # Dataset View Widgets
        self.label = ttk.Label(self.dataset_tab, text="Upload a dataset (CSV) to view the data")
        self.label.pack(pady=10)
        
        self.upload_button = ttk.Button(self.dataset_tab, text="Upload Dataset", command=self.upload_dataset, style="Accent.TButton")
        self.upload_button.pack(pady=10)
        
        # Display Dataset Info Label (for filename, rows, columns)
        self.dataset_info_text = ttk.Label(self.dataset_tab, text="No dataset loaded", font=("Helvetica", 10))
        self.dataset_info_text.pack(pady=10)
        
        # Frame for Treeview and scrollbars
        self.tree_frame = tk.Frame(self.dataset_tab, bg="white")
        self.tree_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Treeview widget with scrollable configuration
        self.tree = ttk.Treeview(self.tree_frame, show="headings", selectmode="browse")
        self.tree.grid(row=0, column=0, sticky="nsew")

        # Vertical scrollbar for Treeview
        self.tree_scroll_y = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree_scroll_y.grid(row=0, column=1, sticky="ns")

        # Horizontal scrollbar for Treeview
        self.tree_scroll_x = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree_scroll_x.grid(row=1, column=0, sticky="ew")

        # Configure Treeview to use the scrollbars
        self.tree.configure(yscrollcommand=self.tree_scroll_y.set, xscrollcommand=self.tree_scroll_x.set)
        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)

        # Row selection binding
        self.tree.bind("<<TreeviewSelect>>", self.display_row_info)

        # Pagination controls at the bottom
        self.pagination_frame = tk.Frame(self.dataset_tab, bg="#F0F0F0")
        self.pagination_frame.pack(pady=10)

        # Previous page button
        self.prev_button = ttk.Button(self.pagination_frame, text="Previous", command=self.previous_page)
        self.prev_button.grid(row=0, column=0, padx=5)
        
        # Page number display
        self.page_label = ttk.Label(self.pagination_frame, text="Page 1")
        self.page_label.grid(row=0, column=1, padx=5)
        
        # Next page button
        self.next_button = ttk.Button(self.pagination_frame, text="Next", command=self.next_page)
        self.next_button.grid(row=0, column=2, padx=5)

        # Rows per page dropdown
        self.rows_per_page_label = ttk.Label(self.pagination_frame, text="Rows per page:")
        self.rows_per_page_label.grid(row=0, column=3, padx=(20, 5))
        
        self.rows_per_page_menu = ttk.Combobox(
            self.pagination_frame, textvariable=self.rows_per_page, values=self.rows_per_page_options, state="readonly", width=5
        )
        self.rows_per_page_menu.grid(row=0, column=4, padx=5)
        self.rows_per_page_menu.bind("<<ComboboxSelected>>", self.update_pagination)

        # Total rows label
        self.total_rows_label = ttk.Label(self.pagination_frame, text="Total rows: 0", font=("Helvetica", 12, "bold"))
        self.total_rows_label.grid(row=0, column=5, padx=(20, 5))

        # Row Details section (unique to Dataset View)
        self.row_info_frame = tk.Frame(self.dataset_tab, bg="#F0F0F0", pady=10)
        self.row_info_frame.pack(fill="x", expand=True)
        self.row_info_label = ttk.Label(self.row_info_frame, text="Row Details:", font=("Helvetica", 13, "bold"))
        self.row_info_label.pack(anchor="w", padx=20)

        self.row_info_scroll = tk.Scrollbar(self.row_info_frame)
        self.row_info_text = tk.Text(self.row_info_frame, wrap="word", height=10, font=("Helvetica", 11), state="disabled", yscrollcommand=self.row_info_scroll.set)
        self.row_info_scroll.pack(side="right", fill="y")
        self.row_info_scroll.config(command=self.row_info_text.yview)
        self.row_info_text.pack(fill="x", padx=20, pady=5)
    
    def setup_analysis_view(self):
        """Sets up the analysis view for text preprocessing with row selection and dataset info display."""
        # Title label
        self.analysis_label = ttk.Label(self.analysis_tab, text="Text Preprocessing", font=("Helvetica", 14, "bold"))
        self.analysis_label.pack(pady=10)

        # Dataset info label (for filename, rows, columns)
        self.dataset_info_text_analysis = ttk.Label(self.analysis_tab, text="No dataset loaded", font=("Helvetica", 10))
        self.dataset_info_text_analysis.pack(pady=10)

        # Row Selection Frame
        row_selection_frame = tk.LabelFrame(self.analysis_tab, text="Row Selection", padx=10, pady=10)
        row_selection_frame.pack(fill="x", padx=20, pady=10)

        # Radio buttons for row selection mode
        self.selection_mode = tk.StringVar(value="slider")
        mode_frame = tk.Frame(row_selection_frame)
        mode_frame.pack(fill="x", pady=(5, 10))

        self.slider_radio = ttk.Radiobutton(mode_frame, text="Use slider", variable=self.selection_mode, value="slider")
        self.slider_radio.grid(row=0, column=0, sticky="w", padx=10)

        self.custom_range_radio = ttk.Radiobutton(mode_frame, text="Use custom range", variable=self.selection_mode, value="range")
        self.custom_range_radio.grid(row=0, column=1, sticky="w", padx=10)

        # Slider and entry for rows to process
        slider_frame = tk.Frame(row_selection_frame)
        slider_frame.pack(fill="x", pady=(5, 10))

        self.rows_to_process = tk.IntVar(value=0)

        # Label for rows to process
        ttk.Label(slider_frame, text="Rows to process:").grid(row=0, column=0, padx=5, sticky="w")

        # Slider for rows to process
        self.row_selection_slider = tk.Scale(
            slider_frame, from_=0, to=10,  # Dynamic range will be updated later
            orient="horizontal", variable=self.rows_to_process, length=250
        )
        self.row_selection_slider.grid(row=0, column=1, padx=(10, 5), pady=(0, 0), sticky="w")

        # Entry for slider value
        self.slider_value_entry = ttk.Entry(slider_frame, textvariable=self.rows_to_process, width=10)
        self.slider_value_entry.grid(row=0, column=2, padx=(10, 0), pady=(8, 0), sticky="w")  # Adjust pady here

        # Bind slider entry to accept manual input
        self.slider_value_entry.bind("<Return>", lambda event: self.update_slider_from_entry())

        # Custom range selection
        custom_range_frame = tk.Frame(row_selection_frame)
        custom_range_frame.pack(fill="x", pady=(5, 10))

        ttk.Label(custom_range_frame, text="Custom range:").grid(row=0, column=0, padx=5, sticky="w")

        self.start_row = tk.IntVar(value=1)
        ttk.Entry(custom_range_frame, textvariable=self.start_row, width=10).grid(row=0, column=1, padx=5, sticky="w")

        ttk.Label(custom_range_frame, text="to").grid(row=0, column=2, padx=5, sticky="w")

        self.end_row = tk.IntVar(value=100)
        ttk.Entry(custom_range_frame, textvariable=self.end_row, width=10).grid(row=0, column=3, padx=5, sticky="w")

        # Preprocessing Options Frame
        preprocess_frame = tk.LabelFrame(self.analysis_tab, text="Preprocessing Options", padx=10, pady=10)
        preprocess_frame.pack(fill="x", padx=20, pady=10)

        # Checkboxes for preprocessing steps
        self.nulls_var = tk.BooleanVar()
        self.lowercase_var = tk.BooleanVar()
        self.stopwords_var = tk.BooleanVar()
        self.stemming_var = tk.BooleanVar()
        self.lemmatization_var = tk.BooleanVar()
        self.punctuation_var = tk.BooleanVar()

        options = [
            ("Remove Nulls", self.nulls_var),
            ("Convert to Lowercase", self.lowercase_var),
            ("Remove Stopwords", self.stopwords_var),
            ("Remove Punctuation", self.punctuation_var),
            ("Apply Stemming", self.stemming_var),
            ("Apply Lemmatization", self.lemmatization_var),
        ]

        for idx, (label, var) in enumerate(options):
            ttk.Checkbutton(preprocess_frame, text=label, variable=var).grid(row=idx, column=0, sticky="w", pady=2)

        # Button to process data
        self.process_button = ttk.Button(self.analysis_tab, text="Generate Cleaned Data", command=self.preprocess_text)
        self.process_button.pack(pady=10)

        # Loading indicator
        self.preprocess_status_label = ttk.Label(self.analysis_tab, text="", font=("Helvetica", 10, "italic"), foreground="gray")
        self.preprocess_status_label.pack(pady=(5, 10))

        # Export button moved to below the process button and aligned to the center
        self.export_button = ttk.Button(self.analysis_tab, text="Export Cleaned Data", command=self.export_cleaned_data)
        self.export_button.pack(pady=10)

        # Frame for displaying processed data with scrollbars
        self.processed_tree_frame = tk.Frame(self.analysis_tab, bg="white")
        self.processed_tree_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.processed_tree = ttk.Treeview(self.processed_tree_frame, show="headings", selectmode="browse")
        self.processed_tree.grid(row=0, column=0, sticky="nsew")

        self.processed_scroll_y = ttk.Scrollbar(self.processed_tree_frame, orient="vertical", command=self.processed_tree.yview)
        self.processed_scroll_y.grid(row=0, column=1, sticky="ns")

        self.processed_scroll_x = ttk.Scrollbar(self.processed_tree_frame, orient="horizontal", command=self.processed_tree.xview)
        self.processed_scroll_x.grid(row=1, column=0, sticky="ew")

        self.processed_tree.configure(yscrollcommand=self.processed_scroll_y.set, xscrollcommand=self.processed_scroll_x.set)
        self.processed_tree_frame.grid_rowconfigure(0, weight=1)
        self.processed_tree_frame.grid_columnconfigure(0, weight=1)

        # Pagination controls for processed data
        self.pagination_frame_analysis = tk.Frame(self.analysis_tab, bg="#F0F0F0")
        self.pagination_frame_analysis.pack(fill="x", padx=20, pady=10)

        self.prev_button_analysis = ttk.Button(self.pagination_frame_analysis, text="Previous", command=self.previous_page_analysis)
        self.prev_button_analysis.grid(row=0, column=0, padx=5)

        self.page_label_analysis = ttk.Label(self.pagination_frame_analysis, text="Page 1")
        self.page_label_analysis.grid(row=0, column=1, padx=5)

        self.next_button_analysis = ttk.Button(self.pagination_frame_analysis, text="Next", command=self.next_page_analysis)
        self.next_button_analysis.grid(row=0, column=2, padx=5)

        # Total rows label for processed data
        self.total_rows_label_analysis = ttk.Label(self.pagination_frame_analysis, text="Total rows: 0", font=("Helvetica", 12, "bold"))
        self.total_rows_label_analysis.grid(row=0, column=3, padx=(20, 5))

    def setup_visualization_view(self):
        """Sets up the visualization tab with a structured layout similar to the vectorization tab."""
        frame = tk.Frame(self.visualization_tab, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        # Title
        ttk.Label(frame, text="Data Visualization", font=("Helvetica", 16, "bold")).pack(pady=(10, 5))

        # Data Selection Frame
        data_selection_frame = ttk.LabelFrame(frame, text="Data Selection", padding=(5, 5))
        data_selection_frame.pack(fill="x", padx=15, pady=(5, 5))

        self.visualize_data_type = tk.StringVar(value="raw")
        ttk.Radiobutton(data_selection_frame, text="Use Raw Data", variable=self.visualize_data_type, value="raw").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(data_selection_frame, text="Use Cleaned Data", variable=self.visualize_data_type, value="cleaned").pack(anchor="w", padx=5, pady=2)

        #Visualization Options Frame
        custom_options_frame = ttk.LabelFrame(frame, text="Visualization Options", padding=(5, 5))
        custom_options_frame.pack(fill="x", padx=15, pady=(5, 5))

        ttk.Label(custom_options_frame, text="Select Categorical Column:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.categorical_column = ttk.Combobox(custom_options_frame, state="enabled", width=30)
        self.categorical_column.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(custom_options_frame, text="Metric:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.metric_type = ttk.Combobox(custom_options_frame, state="enabled", width=20)
        self.metric_type["values"] = ["Count", "Percentage", "Frequency Distribution"]
        self.metric_type.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.metric_type.set("Count")  # Default value

        ttk.Label(custom_options_frame, text="Chart Type:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.chart_type = ttk.Combobox(custom_options_frame, state="enabled", width=30)
        self.chart_type["values"] = [
            "Bar Plot",
            "Pie Chart",
            "Word Cloud",
            "Line Chart",
            "Horizontal Bar Plot",
            "Word Frequency Distribution",
            "Text Length Distribution"
        ]
        self.chart_type.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.chart_type.set("Bar Plot")  # Default value

        # Generate Visualization Button
        self.plot_button = ttk.Button(frame, text="Generate Visualization", command=self.handle_visualization)
        self.plot_button.pack(pady=(10, 10))

        # Main Frame for Plot Display
        plot_frame = tk.Frame(frame)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Plot Display Area
        self.plot_frame = ttk.LabelFrame(plot_frame, text="Visualization", padding=(5, 5))
        self.plot_frame.pack(fill="both", expand=True)

    def upload_dataset(self):
        """Uploads a dataset and initializes it for viewing and analysis."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json")])
        if file_path:
            self.file_path = file_path
            encodings = ['utf-8', 'ISO-8859-1', 'latin1']
            loaded = False
            for enc in encodings:
                try:
                    if file_path.endswith(".csv"):
                        self.dataframe = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
                    elif file_path.endswith(".json"):
                        self.dataframe = pd.read_json(file_path, encoding=enc)
                    loaded = True
                    break
                except Exception:
                    continue

            if not loaded:
                messagebox.showerror("Error", "Failed to load dataset. Please ensure the file is a valid CSV or JSON format.")
                return

            # Limit dataset to 80,000 rows
            max_rows = 50000
            if len(self.dataframe) > max_rows:
                self.dataframe = self.dataframe.iloc[:max_rows]
                messagebox.showwarning(
                    "Dataset Truncated",
                    f"The dataset exceeds {max_rows} rows and has been truncated to the first {max_rows} rows."
                )

            self.notebook.tab(1, state="normal")  # Enable analysis tab after dataset is loaded
            filename = file_path.split("/")[-1]
            filetype = "CSV" if file_path.endswith(".csv") else "JSON"
            dataset_info_text = f"File: {filename} ({filetype}), Features: {len(self.dataframe.columns)}, Rows: {len(self.dataframe)}"
            self.dataset_info_text.config(text=dataset_info_text)
            self.dataset_info_text_analysis.config(text=dataset_info_text)  # Update info on the analysis tab too
            self.total_rows_label.config(text=f"Total rows: {len(self.dataframe)}")  # Update total rows in dataset view

            # Update slider range dynamically
            self.row_selection_slider.config(from_=0, to=len(self.dataframe))  # Set slider range
            self.rows_to_process.set(min(len(self.dataframe), 0))  # Set default slider value to 100 or less
            self.update_pagination()

            # Enable visualization tab and update columns for visualization
            self.update_visualization_columns()
            self.notebook.tab(2, state="normal")  # Enable Visualization tab
            self.notebook.tab(3, state="normal")  # Enable vectorization tab
            self.notebook.tab(4, state="normal")  # Enable Model Application tab
            self.update_model_column_options()

    def update_pagination(self, event=None):
        if self.dataframe is not None:
            self.tree.delete(*self.tree.get_children())
            columns = ["ID"] + list(self.dataframe.columns)
            self.tree["columns"] = columns
            
            for col in columns:
                self.tree.heading(col, text=col, anchor="center")
                self.tree.column(col, anchor="center", minwidth=150, width=200, stretch=True)  # Adjust width as needed

            start_row = self.current_page * self.rows_per_page.get()
            end_row = start_row + self.rows_per_page.get()
            page_data = self.dataframe.iloc[start_row:end_row]

            for idx, row in page_data.iterrows():
                row_number = idx + 1
                row_values = [row_number] + list(row.values)
                self.tree.insert("", "end", values=[str(value) for value in row_values])

            total_pages = (len(self.dataframe) - 1) // self.rows_per_page.get() + 1
            self.page_label.config(text=f"Page {self.current_page + 1} of {total_pages}")
            self.prev_button.config(state="normal" if self.current_page > 0 else "disabled")
            self.next_button.config(state="normal" if end_row < len(self.dataframe) else "disabled")

    def next_page(self):
        self.current_page += 1
        self.update_pagination()

    def previous_page(self):
        self.current_page -= 1
        self.update_pagination()

    def display_row_info(self, event):
        selected_item = self.tree.selection()
        if selected_item:
            row_values = self.tree.item(selected_item)["values"]
            self.row_info_text.config(state="normal")
            self.row_info_text.delete("1.0", tk.END)
            for col, val in zip(self.dataframe.columns, row_values[1:]):
                self.row_info_text.insert(tk.END, f"{col}:\n", "bold")  # Bold column name
                self.row_info_text.insert(tk.END, f"{val}\n\n")  # Regular text for value
            self.row_info_text.tag_configure("bold", font=("Helvetica", 11, "bold"))
            self.row_info_text.config(state="disabled")

    def preprocess_text(self):
        """Processes the data based on selected preprocessing options, optimized for large datasets."""
        if self.dataframe is None:
            messagebox.showerror("Error", "No dataset loaded. Please upload a dataset first.")
            return

        # Ensure at least one preprocessing option is selected
        if not any([
            self.nulls_var.get(),
            self.lowercase_var.get(),
            self.punctuation_var.get(),
            self.stopwords_var.get(),
            self.stemming_var.get(),
            self.lemmatization_var.get()
        ]):
            messagebox.showerror("Error", "Select at least one preprocessing option.")
            return

        # Start preprocessing status
        self.preprocess_status_label.config(text="Processing... Please wait.")
        self.preprocess_status_label.update_idletasks()

        # Define chunk size for batch processing
        CHUNK_SIZE = 1000

        def preprocess_batch(batch):
            """Applies preprocessing to a batch of rows."""
            for col in batch.select_dtypes(include=["object"]).columns:
                if self.punctuation_var.get():
                    batch[col] = batch[col].str.translate(str.maketrans("", "", punctuation))

                if self.lowercase_var.get():
                    batch[col] = batch[col].str.lower()

                if self.stopwords_var.get():
                    stop_words = set(stopwords.words("english"))
                    batch[col] = batch[col].apply(
                        lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words]) if isinstance(x, str) else x
                    )

                if self.stemming_var.get():
                    stemmer = PorterStemmer()
                    batch[col] = batch[col].apply(
                        lambda x: " ".join([stemmer.stem(word) for word in word_tokenize(x)]) if isinstance(x, str) else x
                    )

                if self.lemmatization_var.get():
                    lemmatizer = WordNetLemmatizer()
                    batch[col] = batch[col].apply(
                        lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]) if isinstance(x, str) else x
                    )

            if self.nulls_var.get():
                batch.dropna(inplace=True)

            return batch

        try:
            total_rows = len(self.dataframe)
            processed_chunks = []

            # Process dataset in chunks
            for start_row in range(0, total_rows, CHUNK_SIZE):
                end_row = min(start_row + CHUNK_SIZE, total_rows)
                chunk = self.dataframe.iloc[start_row:end_row].copy()

                # Process the chunk
                processed_chunk = preprocess_batch(chunk)
                processed_chunks.append(processed_chunk)

                # Update progress
                progress = (end_row / total_rows) * 100
                self.preprocess_status_label.config(text=f"Processing... {progress:.2f}% complete")
                self.preprocess_status_label.update_idletasks()

            # Combine processed chunks into a single DataFrame
            self.processed_data = pd.concat(processed_chunks, ignore_index=True)

            # Update UI and enable further steps
            self.update_pagination_analysis()
            self.update_visualization_columns()
            self.total_rows_label_analysis.config(text=f"Total rows: {len(self.processed_data)}")

            self.preprocess_status_label.config(text="Processing complete.")
        except Exception as e:
            self.preprocess_status_label.config(text="Error during preprocessing.")
            messagebox.showerror("Error", f"Preprocessing failed: {e}")

    def display_data(self, dataframe, treeview):
        treeview.delete(*treeview.get_children())
        treeview["columns"] = ["ID"] + list(dataframe.columns)
        
        for col in treeview["columns"]:
            treeview.heading(col, text=col, anchor="center")
            treeview.column(col, anchor="center", width=150)

        for idx, row in dataframe.iterrows():
            row_data = [idx + 1] + list(row)
            treeview.insert("", "end", values=row_data)

    def export_cleaned_data(self):
        if self.processed_data is None or self.processed_data.empty:
            messagebox.showerror("Error", "No cleaned data available to export. Please generate cleaned data first.")
            return

        file_types = [("CSV Files", "*.csv"), ("JSON Files", "*.json")]
        export_path = filedialog.asksaveasfilename(filetypes=file_types)
        if export_path:
            if export_path.endswith(".csv"):
                self.processed_data.to_csv(export_path, index=False)
            elif export_path.endswith(".json"):
                self.processed_data.to_json(export_path, orient="records")
            messagebox.showinfo("Export Success", "Cleaned data exported successfully.")

    def update_pagination_analysis(self):
        if self.processed_data is not None:
            self.processed_tree.delete(*self.processed_tree.get_children())
            columns = ["ID"] + list(self.processed_data.columns)
            self.processed_tree["columns"] = columns

            for col in columns:
                self.processed_tree.heading(col, text=col, anchor="center")
                self.processed_tree.column(col, anchor="center", minwidth=150, width=200, stretch=True)

            start_row = self.current_page * self.rows_per_page.get()
            end_row = start_row + self.rows_per_page.get()
            page_data = self.processed_data.iloc[start_row:end_row]

            for idx, row in page_data.iterrows():
                row_number = idx + 1
                row_values = [row_number] + list(row.values)
                self.processed_tree.insert("", "end", values=[str(value) for value in row_values])

            total_pages = (len(self.processed_data) - 1) // self.rows_per_page.get() + 1
            self.page_label_analysis.config(text=f"Page {self.current_page + 1} of {total_pages}")
            self.prev_button_analysis.config(state="normal" if self.current_page > 0 else "disabled")
            self.next_button_analysis.config(state="normal" if end_row < len(self.processed_data) else "disabled")

    def next_page_analysis(self):
        self.current_page += 1
        self.update_pagination_analysis()

    def previous_page_analysis(self):
        self.current_page -= 1
        self.update_pagination_analysis()

    def update_slider_from_entry(self):
        """Updates the slider value from the manually entered value in the entry field."""
        try:
            value = int(self.slider_value_entry.get())
            max_value = self.row_selection_slider.cget("to")
            if 0 <= value <= max_value:
                self.rows_to_process.set(value)
            else:
                messagebox.showerror("Error", f"Value must be between 0 and {max_value}.")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer.")

    def handle_visualization(self):
        """Handles visualization based on selected type."""
        # Get the dataset type (raw or cleaned)
        data_type = self.visualize_data_type.get()
        data = self.dataframe if data_type == "raw" else self.processed_data

        if data is None or data.empty:
            messagebox.showerror("Error", f"No {data_type} data available for visualization.")
            return

        # Only support custom visualizations
        self.generate_custom_visualization(data)

    def generate_custom_visualization(self, data):
        cat_col = self.categorical_column.get()
        chart = self.chart_type.get()

        if not cat_col:
            messagebox.showerror("Error", "Please select a column for visualization.")
            return

        try:
            # Allow plotting for any column type
            if chart == "Bar Plot":
                grouped_data = data[cat_col].value_counts().head(10)
                fig, ax = plt.subplots()
                grouped_data.plot(kind="bar", ax=ax, color="skyblue")
                ax.set_title(f"Top Categories for {cat_col}")
                ax.set_xlabel(cat_col)
                ax.set_ylabel("Frequency")

            elif chart == "Pie Chart":
                grouped_data = data[cat_col].value_counts().head(10)
                if grouped_data.empty:
                    raise ValueError(f"No valid data found in the column '{cat_col}'.")
                fig, ax = plt.subplots()
                grouped_data.plot.pie(ax=ax, autopct="%1.1f%%", startangle=90)
                ax.set_ylabel("")
                ax.set_title(f"Category Distribution ({cat_col})")

            elif chart == "Horizontal Bar Plot":
                grouped_data = data[cat_col].value_counts().head(10)
                fig, ax = plt.subplots()
                grouped_data.plot(kind="barh", ax=ax, color="skyblue")
                ax.set_title(f"Top Categories for {cat_col}")
                ax.set_xlabel("Frequency")
                ax.set_ylabel(cat_col)

            elif chart == "Line Chart":
                if not pd.api.types.is_datetime64_any_dtype(data[cat_col]):
                    data[cat_col] = pd.to_datetime(data[cat_col], errors="coerce")
                if data[cat_col].isna().all():
                    raise ValueError(f"The column '{cat_col}' cannot be used as a date/time for Line Chart.")
                grouped_data = data.groupby(data[cat_col].dt.to_period("M")).size()
                fig, ax = plt.subplots()
                grouped_data.plot(ax=ax, kind="line", marker="o", color="blue")
                ax.set_title(f"Trend Over Time ({cat_col})")
                ax.set_xlabel("Date")
                ax.set_ylabel("Frequency")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                fig.autofmt_xdate()

            elif chart == "Word Cloud":
                text_data = " ".join(data[cat_col].dropna().astype(str))
                if not text_data.strip():
                    raise ValueError("No valid text data found for Word Cloud.")
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")

            elif chart == "Text Length Distribution":
                text_lengths = data[cat_col].dropna().astype(str).apply(len)
                fig, ax = plt.subplots()
                ax.hist(text_lengths, bins=20, color="skyblue")
                ax.set_title(f"Text Length Distribution ({cat_col})")
                ax.set_xlabel("Text Length")
                ax.set_ylabel("Frequency")

            elif chart == "Word Frequency Distribution":
                text_data = " ".join(data[cat_col].dropna().astype(str))
                if not text_data.strip():
                    raise ValueError("No valid text data found for Word Frequency Distribution.")
                word_tokens = word_tokenize(text_data)
                freq_dist = nltk.FreqDist(word_tokens)
                fig, ax = plt.subplots()
                pd.DataFrame(freq_dist.most_common(20), columns=["Word", "Frequency"]).set_index("Word") \
                    .plot(kind="bar", ax=ax, legend=False, color="skyblue")
                ax.set_title("Word Frequency Distribution")
                ax.set_xlabel("Words")
                ax.set_ylabel("Frequency")

            self.render_plot(fig)

        except ValueError as ve:
            messagebox.showwarning("Visualization Warning", str(ve))
        except Exception as e:
            messagebox.showerror("Visualization Error", f"An error occurred: {e}")

    def update_visualization_columns(self):
        """Update columns for visualization based on the data type."""
        data_type = self.visualize_data_type.get()
        data = self.dataframe if data_type == "raw" else self.processed_data

        if data is not None:
            self.categorical_column["values"] = tuple(data.columns)
            self.categorical_column.set(data.columns[0] if len(data.columns) > 0 else "")

    def render_plot(self, figure):
        """Renders a matplotlib figure in the plot_frame."""
        # Clear the existing plot, if any
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Add the new figure
        canvas = FigureCanvasTkAgg(figure, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_feedback(self, message):
        """Displays a message in the visualization frame."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        feedback_label = ttk.Label(self.plot_frame, text=message, font=("Helvetica", 14), anchor="center", justify="center")
        feedback_label.pack(expand=True)

    def apply_vectorization(self):
        """Applies text vectorization, displays the result, and generates corresponding charts."""
        data_source = self.vectorize_data_source.get()
        data = self.processed_data if data_source == "processed" else self.dataframe

        if data is None or data.empty:
            messagebox.showerror("Error", "No data available. Please preprocess or upload a dataset first.")
            return

        # Automatically detect text columns
        text_columns = [
            col for col in data.columns
            if data[col].dropna().apply(lambda x: isinstance(x, str)).all()
        ]

        if not text_columns:
            messagebox.showerror(
                "Error",
                "No text columns available for vectorization. Ensure your dataset contains valid text data."
            )
            return

        # Combine text columns into a single text column for vectorization
        data["Combined_Text"] = data[text_columns].fillna("").apply(lambda row: " ".join(row), axis=1)
        text_data = data["Combined_Text"]

        try:
            method = self.vectorization_method.get()

            self.vectorization_output.config(state="normal")
            self.vectorization_output.delete("1.0", tk.END)
            self.vectorization_output.insert("1.0", "Vectorizing...")
            self.vectorization_output.config(state="disabled")
            self.vectorization_output.update_idletasks()

            if method == "pos_ner":
                try:
                    max_rows = 1000
                    if len(text_data) > max_rows:
                        text_data = text_data.head(max_rows)
                        self.update_output("Processing limited to the first 1000 rows due to dataset size constraints.")

                    docs = nlp.pipe(text_data, disable=["tagger", "parser"], batch_size=100)
                    named_entities = [{ent.text: ent.label_ for ent in doc.ents} for doc in docs]

                    # Group by entity type
                    entity_type_counts = {}
                    for doc_ents in named_entities:
                        for label in doc_ents.values():
                            entity_type_counts[label] = entity_type_counts.get(label, 0) + 1

                    # Sort by frequency and get the top 10
                    sorted_entities = sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True)
                    top_entities = sorted_entities[:10]

                    # Display summarized results
                    output_text = "NER applied successfully.\n\nTop Entity Types:\n"
                    output_text += "\n".join([f" - {label}: {count}" for label, count in top_entities])
                    self.update_output(output_text)

                    # Generate visualization
                    labels, counts = zip(*top_entities)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(labels, counts, color="skyblue")
                    ax.set_title("Top 10 Named Entity Types")
                    ax.set_xlabel("Entity Type")
                    ax.set_ylabel("Frequency")
                    plt.tight_layout()
                    self.display_chart(fig)

                    # Update vectorized matrix
                    ner_df = pd.DataFrame({"Text": text_data, "Named Entities": named_entities})
                    self.display_matrix(ner_df)

                except Exception as e:
                    self.update_output(f"Error during NER: {e}")

            elif method == "ngrams":
                try:
                    from sklearn.feature_extraction.text import CountVectorizer
                    # Ensure text data is valid
                    text_data = text_data.dropna().astype(str).tolist()

                    # Apply CountVectorizer
                    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=100, stop_words='english')
                    ngram_matrix = vectorizer.fit_transform(text_data)
                    ngram_df = pd.DataFrame(ngram_matrix.toarray(), columns=vectorizer.get_feature_names_out())
                    self.display_matrix(ngram_df)

                    # Generate N-Grams Chart
                    ngram_freq = ngram_df.sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ngram_freq.head(10).plot(kind="bar", ax=ax, color="skyblue")
                    ax.set_title("Top N-Grams Frequencies")
                    ax.set_ylabel("Frequency")
                    ax.set_xlabel("N-Grams")
                    plt.tight_layout()
                    self.display_chart(fig)

                    # Update vectorization output
                    self.update_output(
                        "N-Grams vectorization applied successfully.\n\n"
                        "Top 10 N-Grams and their frequencies:\n" +
                        "\n".join([f" - {ngram}: {freq}" for ngram, freq in ngram_freq.head(10).items()])
                    )
                except Exception as e:
                    self.update_output(f"Error during N-Grams vectorization: {e}")

            elif method == "tfidf":
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=10)
                self.vectorized_data = vectorizer.fit_transform(text_data)
                feature_names = vectorizer.get_feature_names_out()
                vectorized_df = pd.DataFrame(self.vectorized_data.toarray(), columns=feature_names)
                self.display_matrix(vectorized_df)

                # Generate TF-IDF Chart
                mean_scores = vectorized_df.mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6, 4))
                mean_scores.plot(kind="bar", ax=ax, color="skyblue")
                ax.set_title("Average TF-IDF Scores for Top Features")
                ax.set_ylabel("Score")
                ax.set_xlabel("Feature Names")
                ax.set_xticklabels(mean_scores.index, rotation=45)
                plt.tight_layout()
                self.display_chart(fig)

                self.update_output(
                    "**TF-IDF Vectorization**\n\n"
                    "TF-IDF identifies important words in the text.\n"
                    "Below are the top features:\n\n" +
                    "\n".join(f" - Feature {i+1}: '{feature}'" for i, feature in enumerate(feature_names))
                )

            elif method == "word2vec":
                from gensim.models import Word2Vec
                tokenized_data = [word_tokenize(sentence) for sentence in text_data]
                model = Word2Vec(sentences=tokenized_data, vector_size=10, window=5, min_count=1, workers=4)
                words = list(model.wv.index_to_key)[:10]
                vectors = [model.wv[word][:3] for word in words]
                vectorized_df = pd.DataFrame(vectors, index=words, columns=["Dim1", "Dim2", "Dim3"])
                self.display_matrix(vectorized_df)

                # Generate Word2Vec Chart
                mean_scores = vectorized_df.mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6, 4))
                mean_scores.plot(kind="bar", ax=ax, color="skyblue")
                ax.set_title("Average Word2Vec Scores for Top Dimensions")
                ax.set_ylabel("Score")
                ax.set_xlabel("Dimensions")
                ax.set_xticklabels(mean_scores.index, rotation=45)
                plt.tight_layout()
                self.display_chart(fig)

                self.update_output(
                    "**Word2Vec Vectorization**\n\n"
                    "Word embeddings for significant words (top 3 dimensions):\n\n" +
                    "\n".join(f" - Word '{word}': Dim1={vec[0]:.4f}, Dim2={vec[1]:.4f}, Dim3={vec[2]:.4f}"
                            for word, vec in zip(words, vectors))
                )

            elif method == "bow":
                from sklearn.feature_extraction.text import CountVectorizer
                vectorizer = CountVectorizer(max_features=10)
                self.vectorized_data = vectorizer.fit_transform(text_data)
                feature_names = vectorizer.get_feature_names_out()
                vectorized_df = pd.DataFrame(self.vectorized_data.toarray(), columns=feature_names)
                self.display_matrix(vectorized_df)

                # Generating the Bag of Words Distribution
                term_frequencies = vectorized_df.sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6, 4))
                term_frequencies.plot(kind="bar", ax=ax, color="skyblue")
                ax.set_title("Bag of Words Term Frequencies")
                ax.set_ylabel("Frequency")
                ax.set_xlabel("Terms")
                ax.set_xticklabels(term_frequencies.index, rotation=45)
                plt.tight_layout()
                self.display_chart(fig)

                self.update_output(
                    "**Bag of Words Vectorization**\n\n"
                    "Top terms and their frequencies:\n\n" +
                    "\n".join(f" - Term '{term}': Frequency={freq}" for term, freq in zip(feature_names, vectorized_df.sum(axis=0)))
                )

            self.notebook.tab(4, state="normal")
            messagebox.showinfo("Info", "Vectorization complete. Data ready for modeling.")

        except Exception as e:
            self.vectorization_output.config(state="normal")
            self.vectorization_output.delete("1.0", tk.END)
            self.vectorization_output.insert("1.0", f"Error during vectorization: {e}")
            self.vectorization_output.config(state="disabled")
    
    def process_chunk(self, chunk):
        """Process a chunk of data for POS tagging and NER."""
        def extract_named_entities(tree):
            return [" ".join(c[0] for c in child) for child in tree if isinstance(child, Tree)]

        pos_tags = [pos_tag(word_tokenize(sentence)) for sentence in chunk]
        named_entities = [extract_named_entities(ne_chunk(tagged)) for tagged in pos_tags]
        return pd.DataFrame({"Text": chunk, "POS Tags": pos_tags, "Named Entities": named_entities})
    
    def update_output(self, text):
        """Helper function to update the vectorization output."""
        self.vectorization_output.config(state="normal")
        self.vectorization_output.delete("1.0", tk.END)
        self.vectorization_output.insert("1.0", text)
        self.vectorization_output.config(state="disabled")

    def display_chart(self, figure):
        """Displays the chart in the chart frame of the vectorization tab."""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(figure, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_matrix(self, dataframe):
        """Displays the vectorized matrix in a scrollable table with vertical scrolling only."""
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        matrix_tree = ttk.Treeview(self.matrix_frame, show="headings")
        matrix_tree["columns"] = ["Index"] + list(dataframe.columns)
        matrix_tree.heading("Index", text="Index")
        for col in dataframe.columns:
            matrix_tree.heading(col, text=col)
            matrix_tree.column(col, anchor="center", width=100)  # Adjust width as needed

        for idx, row in dataframe.iterrows():
            matrix_tree.insert("", "end", values=[idx] + list(row))

        # Add vertical scrollbar only
        scroll_y = ttk.Scrollbar(self.matrix_frame, orient="vertical", command=matrix_tree.yview)
        matrix_tree.configure(yscrollcommand=scroll_y.set)

        # Pack treeview and vertical scrollbar
        matrix_tree.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")

    def display_tfidf_chart(self, dataframe):
        """Displays a bar chart of TF-IDF scores."""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        mean_scores = dataframe.mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))  # Adjust size as needed
        mean_scores.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Average TF-IDF Scores for Top Features")
        ax.set_ylabel("Score")
        ax.set_xlabel("Feature Names")
        ax.set_xticklabels(mean_scores.index, rotation=45)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_vectorization_view(self):
        """Sets up the vectorization tab for text preprocessing."""
        frame = tk.Frame(self.vectorization_tab, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        # Title
        ttk.Label(frame, text="Text Representation", font=("Helvetica", 16, "bold")).pack(pady=(10, 5))

        # Explanation Label
        explanation = (
            "TF-IDF: Highlights important terms based on their frequency in the dataset.\n"
            "Word2Vec: Creates dense word embeddings that capture semantic meanings."
        )
        explanation_label = tk.Text(
            frame, wrap="word", height=3, font=("Helvetica", 12), bg="#F0F0F0", relief="flat"
        )
        explanation_label.insert("1.0", explanation)

        # Center alignment for the text
        explanation_label.tag_configure("center", justify="center")
        explanation_label.tag_add("center", "1.0", "end")

        # Bold formatting for specific terms
        explanation_label.tag_configure("bold", font=("Helvetica", 12, "bold"))
        explanation_label.tag_add("bold", "1.0", "1.6")  # Bold "TF-IDF"
        explanation_label.tag_add("bold", "2.0", "2.8")  # Bold "Word2Vec"

        explanation_label.config(state="disabled")
        explanation_label.pack(pady=(5, 5))

        # Data Selection for Vectorization
        data_selection_frame = ttk.LabelFrame(frame, text="Data Selection", padding=(5, 5))
        data_selection_frame.pack(fill="x", padx=15, pady=(5, 5))

        self.vectorize_data_source = tk.StringVar(value="processed")
        ttk.Radiobutton(data_selection_frame, text="Use Cleaned Data", variable=self.vectorize_data_source, value="processed").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(data_selection_frame, text="Use Raw Data", variable=self.vectorize_data_source, value="raw").pack(anchor="w", padx=5, pady=2)

        # Vectorization Options
        options_frame = ttk.LabelFrame(frame, text="Vectorization Options", padding=(5, 5))
        options_frame.pack(fill="x", padx=15, pady=(5, 5))

        self.vectorization_method = tk.StringVar(value="tfidf")
        ttk.Radiobutton(options_frame, text="TF-IDF (Term Frequency - Inverse Document Frequency)",
                        variable=self.vectorization_method, value="tfidf").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(options_frame, text="Word2Vec (Word Embeddings)",
                        variable=self.vectorization_method, value="word2vec").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(options_frame, text="Bag of Words (Term Frequency)",
                        variable=self.vectorization_method, value="bow").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(options_frame, text="NER (Named Entity Recognition)",
                        variable=self.vectorization_method, value="pos_ner").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(options_frame, text="N-Grams (Frequent Word Combinations)",
                        variable=self.vectorization_method, value="ngrams").pack(anchor="w", padx=5, pady=2)

        # Apply Vectorization Button
        self.vectorize_button = ttk.Button(frame, text="Apply Vectorization", command=self.apply_vectorization)
        self.vectorize_button.pack(pady=(5, 10))

        # Main Frame for Output, Matrix, and Chart
        main_frame = tk.Frame(frame)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left Frame for Vectorization Output
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Right Frame for Matrix and Chart
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)

        # Vectorization Output as Scrollable Text
        results_frame = ttk.LabelFrame(left_frame, text="Vectorization Output", padding=(5, 5))
        results_frame.pack(fill="both", expand=True)

        self.vectorization_output = tk.Text(results_frame, wrap="word", height=15, state="disabled", bg="#f9f9f9", font=("Consolas", 10))
        self.vectorization_output.pack(fill="both", expand=True)

        # Add scrollbar for vectorization output
        output_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.vectorization_output.yview)
        self.vectorization_output.configure(yscrollcommand=output_scroll.set)
        output_scroll.pack(side="right", fill="y")

        # Matrix and Chart Frames
        self.matrix_frame = ttk.LabelFrame(right_frame, text="Vectorized Matrix", padding=(5, 5))
        self.matrix_frame.pack(fill="both", expand=True, pady=(0, 5))

        self.chart_frame = ttk.LabelFrame(right_frame, text="Chart", padding=(5, 5))
        self.chart_frame.pack(fill="both", expand=True)  # Maximized vertical space for the chart

    def setup_model_application_view(self):
        """Sets up the model application tab for training and evaluating classification models."""
        frame = tk.Frame(self.model_application_tab, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        # Title
        ttk.Label(frame, text="Model Application", font=("Helvetica", 16, "bold")).pack(pady=(10, 5))

        # Data Selection Frame
        data_selection_frame = ttk.LabelFrame(frame, text="Data Selection", padding=(5, 5))
        data_selection_frame.pack(fill="x", padx=15, pady=(5, 5))

        self.feature_column = tk.StringVar(value="")
        self.label_column = tk.StringVar(value="")

        ttk.Label(data_selection_frame, text="Select Text Column (Feature):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.feature_dropdown = ttk.Combobox(data_selection_frame, textvariable=self.feature_column, state="readonly", width=30)
        self.feature_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(data_selection_frame, text="Select Label Column:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.label_dropdown = ttk.Combobox(data_selection_frame, textvariable=self.label_column, state="readonly", width=30)
        self.label_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Model Selection Frame
        model_selection_frame = ttk.LabelFrame(frame, text="Model Selection", padding=(5, 5))
        model_selection_frame.pack(fill="x", padx=15, pady=(5, 5))

        self.selected_model = tk.StringVar(value="Naive Bayes")
        ttk.Radiobutton(model_selection_frame, text="Naive Bayes", variable=self.selected_model, value="Naive Bayes").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(model_selection_frame, text="Logistic Regression", variable=self.selected_model, value="Logistic Regression").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(model_selection_frame, text="Decision Tree", variable=self.selected_model, value="Decision Tree").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(model_selection_frame, text="Support Vector Machine (SVM)", variable=self.selected_model, value="SVM").pack(anchor="w", padx=5, pady=2)
        ttk.Radiobutton(model_selection_frame, text="K-Nearest Neighbors (KNN)", variable=self.selected_model, value="KNN").pack(anchor="w", padx=5, pady=2)

        # Model Parameters Frame
        model_frame = ttk.LabelFrame(frame, text="Model Parameters", padding=(5, 5))
        model_frame.pack(fill="x", padx=15, pady=(5, 5))

        # Test Size as Percentage
        self.test_size_percentage = tk.IntVar(value=30)  # Default is 30%
        ttk.Label(model_frame, text="Test Size (Percentage):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(model_frame, textvariable=self.test_size_percentage, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Random State as Boolean
        self.use_static_split = tk.BooleanVar(value=True)  # Default to consistent split
        ttk.Checkbutton(model_frame, text="Ensure Consistent Results (Static Split)", variable=self.use_static_split).grid(
            row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w"
        )

        # Apply Model Button
        self.apply_model_button = ttk.Button(model_frame, text="Apply Model", command=self.apply_model)
        self.apply_model_button.grid(row=2, column=0, columnspan=2, pady=(10, 5))

        # Output Frame
        self.model_output_frame = ttk.LabelFrame(frame, text="Model Output", padding=(5, 5))
        self.model_output_frame.pack(fill="both", expand=True, padx=15, pady=(10, 5))

        self.model_output_text = tk.Text(self.model_output_frame, wrap="word", height=10, state="disabled", bg="#f9f9f9", font=("Consolas", 10))
        self.model_output_text.pack(fill="both", expand=True)

    def apply_model(self):
        """Applies the selected model and displays the accuracy score along with additional insights."""
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.preprocessing import LabelEncoder
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        if self.processed_data is None or self.vectorized_data is None:
            messagebox.showerror("Error", "Cleaned and vectorized data is required for model application. Please preprocess and vectorize the data first.")
            return

        feature_column = self.feature_column.get()
        label_column = self.label_column.get()

        if not feature_column or not label_column:
            messagebox.showerror("Error", "Please select a text column (feature) and label column before applying a model.")
            return

        try:
            # Clear existing widgets in model output frame
            for widget in self.model_output_frame.winfo_children():
                widget.destroy()

            # Recreate model output layout
            self.model_output_text = tk.Text(
                self.model_output_frame,
                wrap="word",
                height=15,
                state="disabled",
                bg="#1E1E1E",
                fg="#FFFFFF",
                font=("Consolas", 12),
                relief="flat",
            )
            self.model_output_text.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=5)

            plot_frame = tk.Frame(self.model_output_frame)
            plot_frame.pack(side="right", fill="both", expand=False, padx=(5, 10), pady=5)

            # Prepare data
            X = self.vectorized_data  # Assume vectorized data is ready
            y = self.processed_data[label_column]

            # Ensure the label column is numeric
            if not pd.api.types.is_numeric_dtype(y):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
            else:
                class_mapping = {i: i for i in y.unique()}

            # Convert test size to a fraction
            test_size = self.test_size_percentage.get() / 100.0
            random_state = 42 if self.use_static_split.get() else None

            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )

            # Select model
            model_name = self.selected_model.get()
            if model_name == "Naive Bayes":
                model = MultinomialNB()
            elif model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "SVM":
                model = SVC()
            elif model_name == "KNN":
                model = KNeighborsClassifier()
            else:
                messagebox.showerror("Error", f"Unknown model: {model_name}")
                return

            # Train model
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_labels = [class_mapping[i] for i in sorted(class_mapping.keys())]

            # Prepare model evaluation text with enhanced formatting
            evaluation_text = (
                f"\n{'='*40}\n"
                f"{'MODEL EVALUATION':^40}\n"
                f"{'='*40}\n\n"
                f"Model Name: {model_name}\n"
                f"Accuracy  : {accuracy:.2f}\n\n"
                f"{'Classification Report':^40}\n"
                f"{'-'*40}\n"
            )

            for label_idx, metrics in report.items():
                if isinstance(metrics, dict):  # Skip average values
                    label = cm_labels[int(label_idx)] if label_idx.isdigit() else label_idx
                    evaluation_text += (
                        f"{label:<15}: Precision={metrics['precision']:.2f}, "
                        f"Recall={metrics['recall']:.2f}, F1-Score={metrics['f1-score']:.2f}\n"
                    )

            evaluation_text += (
                f"\n{'OBSERVATIONS':^40}\n"
                f"{'-'*40}\n"
                f"- Accuracy of {accuracy:.2f} suggests the model performs reasonably well.\n"
                f"- Logistic Regression consistently outperforms Naive Bayes for text classification tasks due to its ability to handle feature correlations.\n"
                f"- Areas with high false positives/negatives in the confusion matrix indicate data imbalance.\n"
                f"- Fine-tuning hyperparameters or using cross-validation could yield more robust results.\n"
            )

            # Display model evaluation text with styles
            self.model_output_text.config(state="normal")
            self.model_output_text.delete("1.0", tk.END)
            self.model_output_text.insert("1.0", evaluation_text)
            self.model_output_text.tag_add("center", "1.0", "end")
            self.model_output_text.config(state="disabled")

            # Plot confusion matrix with labels
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=cm_labels, yticklabels=cm_labels, ax=ax)

            ax.set_title("Confusion Matrix", fontsize=14, color="#333333")
            ax.set_xlabel("Predicted Label", fontsize=12, color="#333333")
            ax.set_ylabel("True Label", fontsize=12, color="#333333")

            # Render plot in plot frame
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply model: {e}")

    def update_model_column_options(self):
        """Updates dropdown options for feature and label columns."""
        if self.dataframe is None:
            # If no data is loaded, clear the dropdowns
            self.feature_dropdown["values"] = []
            self.feature_column.set("")
            self.label_dropdown["values"] = []
            self.label_column.set("")
            return

        # Identify potential feature columns (string-based or high-cardinality columns)
        potential_features = [
            col for col in self.dataframe.columns
            if self.dataframe[col].dtype == object or self.dataframe[col].nunique() > 10  # Assume features are high-cardinality
        ]

        # Identify potential label columns (categorical by nature)
        potential_labels = [
            col for col in self.dataframe.columns
            if self.dataframe[col].nunique() <= 10  # Low cardinality indicates categorical
            and not pd.api.types.is_float_dtype(self.dataframe[col])  # Avoid pure numeric floats
        ]

        # Update feature and label dropdowns
        self.feature_dropdown["values"] = potential_features
        self.feature_column.set(potential_features[0] if potential_features else "")

        self.label_dropdown["values"] = potential_labels
        self.label_column.set(potential_labels[0] if potential_labels else "")

# Main application
root = tk.Tk()
app = DataAnalyzerApp(root)
root.mainloop()