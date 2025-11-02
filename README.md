# -Feature-Engineering-



# My Machine Learning Feature Engineering Project

This is my project for my data science assignment. The goal wasn't just to build a model, but to really focus on all the **data preparation steps** that happen *before* you even start training.

I wanted to show how these steps change depending on the data, so I worked with two completely different datasets:
1.  **Image Data:** The classic "Dog vs. Cat" dataset.
2.  **Text Data:** 50,000 IMDB Movie Reviews.

I did all the work in a Google Colab notebook using Python, mainly with TensorFlow, Pandas, and Scikit-learn.

---

## ⚙️ What I Did: The Feature Engineering Steps

### 1. For the Image Data (Dog vs. Cat) 🖼️

Working with images is very different from a simple table, so I used a modern deep learning approach.

* **Loading & Labeling:** I used a handy TensorFlow function (`image_dataset_from_directory`) that did two jobs at once. It loaded all the images *and* automatically handled the labels. It saw the 'cat' and 'dog' folder names and just knew to label them as `0` and `1`.
* **Scaling the Pixels:** Neural networks work best when input numbers are small. Instead of the usual `[0, 255]` pixel values, I used a special function (`preprocess_input`) to scale them all to the `[-1, 1]` range. This helps the model train much faster.
* **Extracting Features (The Cool Part):** This was the most important step. Instead of training a model from scratch (which takes forever), I used **Transfer Learning**. I loaded a powerful, pre-trained model called `MobileNetV2` and "froze" it. I basically used it as a super-smart feature extractor. It already knows how to find edges, textures, and shapes, so it did all the heavy lifting for me.
* **The Result:** It worked incredibly well! By just adding a tiny classifier on top of these pre-trained features, my model got **100% accuracy** on the validation set. It really shows how powerful this technique is.

### 2. For the Text Data (IMDB Reviews) 📝

This was a more "classic" NLP problem, so I built a pipeline using Pandas and Scikit-learn.

* **Cleaning the Text:** This was the biggest job. The raw reviews were messy—full of HTML tags (`<br />`), punctuation, and common "stop words" (like 'the', 'is', 'a') that just add noise. I wrote a function to strip all of that junk out and turn everything into lowercase.
* **Encoding Labels:** Just like with the images, I had to turn the text labels ("positive" and "negative") into numbers. `LabelEncoder` made this easy, converting them to `1` and `0`.
* **Turning Words into Numbers (TF-IDF):** A model can't read words, so I used `TfidfVectorizer`. This tool is smart: it converts all the cleaned reviews into a big matrix of numbers, scoring words based on how *important* they are to a review (not just how often they appear). I had it find the 5,000 most important words.
* **Selecting the *Best* Features:** 5,000 features is still a lot. So, I used a statistical test (`chi2`) to find the **top 2,000 features** from that list that were the *most* predictive of a good or bad review.
* **The Result:** This pipeline turned a huge, messy pile of text into a clean, smart, and efficient set of features. A simple Naive Bayes model trained on this final data got **~88% accuracy**, which shows the whole process was a success!

---

##  A Note on Ethics

The assignment also asked about the ethics of using features like 'Gender' or 'Marital Status' to predict something like employee attrition.

My conclusion is that it's a terrible idea. It's a textbook example of bias, where the model would just learn to discriminate based on historical prejudices. The best way to mitigate this is simple: **don't use those features.** The next step would be to actively check if other features (like 'Job Title' or 'Department') are acting as a secret proxy for them.
