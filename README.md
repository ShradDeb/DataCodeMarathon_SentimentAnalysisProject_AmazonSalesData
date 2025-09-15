# Amazon Product Review Sentiment Analysis

This project is a comprehensive **Natural Language Processing (NLP)** and machine learning project that analyzes customer reviews from Amazon. The primary goal is to build a sentiment analysis model to classify reviews as positive, negative, or neutral. This provides valuable insights into customer satisfaction and product performance.

##  Project Highlights

- **Data Source:** A real-world dataset of Amazon product reviews, including review titles, content, and star ratings. (https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset/code)
- **NLP Techniques:** Applied advanced text preprocessing techniques to clean and normalize the review text.
- **Machine Learning Model:** Developed a sentiment classification model using a **Multinomial Naive Bayes** classifier.
- **Actionable Insights:** The model can be used to quickly gauge public opinion on products, identify areas for improvement, and monitor brand reputation.
- - **Interactive Visualizations:** Used Plotly Express to create dynamic and interactive charts, allowing for a deeper exploration of the data.

##  Tools and Libraries Used

* **Python:** The core programming language.
* **Pandas:** For data manipulation and analysis.
* **NLTK (Natural Language Toolkit):** For text preprocessing, including tokenization, lemmatization, and stop word removal.
* **Scikit-learn:** For machine learning tasks, including TF-IDF vectorization, model training, and evaluation.
* **Plotly Express:** To create interactive and visually appealing data visualizations.
* **Matplotlib & WordCloud:** Used to generate word clouds for each sentiment category.
* **Jupyter Notebook/VS Code:** For an interactive coding environment.

##  Project Structure

* **amazon.csv:** The dataset used for this project.
* **sentiment_analysis.py or sentiment_analysis.ipynb:** The main script/notebook containing all the code for data preprocessing, model training, and visualization.
* **README.md:** This file, providing an overview of the project.

##  Methodology

### 1. Data Preprocessing
The raw review text was a mix of titles and content, containing special characters, numbers, and extra spaces. The following steps were performed to prepare the data for analysis:
- Combined review_title and review_content into a single text column.
- Cleaned the text by removing non-alphabetic characters and converting it to lowercase.
- Preprocessed the cleaned text by removing common English stop words (e.g., "the", "a", "is") and applying lemmatization to reduce words to their base form (e.g., "running" to "run").

### 2. Sentiment Labeling
The `rating` column was used to create sentiment labels:
- **Positive:** Reviews with a rating of 3.5 or higher.
- **Negative:** Reviews with a rating of 2.5 or lower.
- **Neutral:** Reviews with a rating of 3.

### 3. Feature Extraction
The processed text data was converted into a numerical format using **TF-IDF (Term Frequency-Inverse Document Frequency)**. This technique weighs words based on their importance in the document and the entire dataset, allowing the machine learning model to understand the context.

### 4. Data Visualization
Key insights were visualized using Plotly Express to create interactive bar charts showing:
   - The distribution of product ratings.
   - The breakdown of sentiment labels (Positive, Negative, Neutral).
   - The top 10 most reviewed product categories.
A word cloud for each sentiment was also generated to highlight the most common words associated with each sentiment category.


### 5. Model Training and Evaluation
A **Multinomial Naive Bayes** classifier was trained on the TF-IDF vectors. The model's performance was evaluated using standard metrics like **accuracy** and a **classification report**, which provides precision, recall, and F1-score for each sentiment class.

## Results

The model achieved a strong performance on the test set, demonstrating its ability to effectively classify the sentiment of unseen reviews. The interactive visualizations provide a clear, exploratory view of the dataset, from overall rating trends to specific category performance.

## How to Run the Code

1.  Clone this repository to your local machine.
2.  Ensure you have Python installed.
3.  Install the required libraries:
    ```
    pip install pandas scikit-learn nltk
    ```
4.  Place the `amazon.csv` file in the same directory as the script.
5.  Run the script from your terminal:
    ```
    python sentiment_analysis.py
    ```

Feel free to explore the code, modify the parameters, and experiment with different models!

---

**Author:** **Shraddha Debata**

**Connect with me on LinkedIn:**  https://www.linkedin.com/in/shraddha-debata-59726094

**View more of my work:** 
          **Tableau** https://public.tableau.com/app/profile/shraddha.debata2941/vizzes 
          **Kaggle** https://www.kaggle.com/code/shraddhadebata/notebooke11b1b1723
