import pandas as pd # Used for working with data in tables (like Excel spreadsheets)
import matplotlib.pyplot as plt # Used for creating static, interactive, and animated visualizations
import seaborn as sns # Built on matplotlib, provides a high-level interface for drawing attractive statistical graphics
from sklearn.model_selection import train_test_split # Used to split data into training and testing sets
from sklearn.preprocessing import StandardScaler # Used to scale (standardize) our data, important for KNN
from sklearn.neighbors import KNeighborsClassifier # This is our K-Nearest Neighbors algorithm!
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # Used to evaluate how well our model performed
import numpy as np # Used for numerical operations, especially with arrays

# --- 1. Data Loading and Preparation (Getting Our Data Ready) ---

# Load the dataset from the uploaded CSV file.
# Think of 'df' as your entire hotel reservation spreadsheet.
df = pd.read_csv('first inten project.csv')

print("--- Initial Dataset Information ---")
print("Dataset Info:")
print(df.info()) # Shows a summary of our data: how many rows, columns, and what type of data is in each column.
print("\nFirst 5 rows of the Dataset:")
print(df.head()) # Shows the very first few rows of our data, so we can see what it looks like.
print("\nAll Column Names Before Cleaning:")
print(df.columns.tolist()) # Lists all the original column names in our spreadsheet.

# Clean column names:
# We're making column names easier to work with in Python.
# 1. .str.strip(): Removes any extra spaces from the beginning or end of column names.
# 2. .str.lower(): Converts all column names to lowercase (e.g., 'LeadTime' becomes 'leadtime').
# 3. .str.replace(' ', ''): Replaces spaces between words with no space (e.g., 'lead time' becomes 'leadtime').
# 4. .str.replace('[^a-z0-9]', '', regex=True): Removes any special characters (like hyphens or parentheses)
#    that are not letters, numbers, or underscores.
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '').str.replace('[^a-z0-9]', '', regex=True)

print("\nAll Column Names After Cleaning:")
print(df.columns.tolist()) # Lists the column names after our cleaning process.

# Rename columns to standardized, more descriptive names.
# We're giving some columns new, clearer names that match our problem description.
# For example, 'bookingstatus' is now 'is_canceled' because that's what it represents.
# FIX: Use the actual cleaned column names for renaming.
df.rename(columns={
    'bookingstatus': 'is_canceled', # Our main goal: predicting if a booking is 'canceled' or 'not_canceled'
    'averageprice': 'adr', # 'ADR' stands for Average Daily Rate, essentially the price.
    'repeated': 'is_repeated_guest', # Tells us if the guest has stayed before.
    'pc': 'previous_cancellations' # 'P-C' from the original file, means previous cancellations.
}, inplace=True) # 'inplace=True' means we apply these changes directly to our 'df' spreadsheet.

print("\nAll Column Names After Renaming for Modeling:")
print(df.columns.tolist()) # Lists the column names after we've renamed them.

# Convert the 'is_canceled' column (our target) into numbers.
# Machine learning models work best with numbers. So, we change 'Canceled' to 1 and 'Not_Canceled' to 0.
# IMPORTANT FIX: Convert to lowercase before comparison to handle 'Canceled' vs 'canceled'.
df['is_canceled'] = df['is_canceled'].apply(lambda x: 1 if str(x).lower() == 'canceled' else 0)

print("\nValue counts for 'is_canceled' after conversion:")
# This shows us how many bookings were canceled (1) and how many were not (0).
print(df['is_canceled'].value_counts())

# --- 2. Exploratory Data Analysis (EDA) and Visualization (Understanding Our Data with Pictures) ---
# Set up the style for our plots to make them look nice and consistent.
sns.set_style("whitegrid")

# Create and save plots to help us see relationships between different pieces of information
# and whether a booking was canceled.

# 1. Lead Time vs. Cancellation
# 'Lead Time' is the number of days between the booking date and the arrival date.
# We use a 'boxplot' to see the distribution of lead times for canceled (1) vs. not canceled (0) bookings.
plt.figure(figsize=(10, 6)) # Sets the size of our plot.
sns.boxplot(x='is_canceled', y='leadtime', data=df) # Use 'leadtime' (cleaned name)
plt.title('Lead Time vs. Cancellation (0 = Not Canceled, 1 = Canceled)', fontsize=14) # Title of the plot.
plt.xlabel('Is Canceled', fontsize=12) # Label for the X-axis.
plt.ylabel('Lead Time (Days)', fontsize=12) # Label for the Y-axis.
plt.xticks(ticks=[0, 1], labels=['Not Canceled', 'Canceled']) # Custom labels for 0 and 1 on the X-axis.
plt.grid(axis='y', linestyle='--', alpha=0.7) # Adds a light grid for better readability.
plt.tight_layout() # Adjusts plot parameters for a tight layout.
plt.savefig('lead_time_vs_cancellation.png') # Saves the plot as an image file.
plt.close() # Closes the plot to free up memory.

# 2. ADR (Price) vs. Cancellation
# 'ADR' is the Average Daily Rate, which is essentially the price of the room.
# Another box plot to see how price relates to cancellation.
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_canceled', y='adr', data=df)
plt.title('ADR (Average Daily Rate) vs. Cancellation (0 = Not Canceled, 1 = Canceled)', fontsize=14)
plt.xlabel('Is Canceled', fontsize=12)
plt.ylabel('ADR (Average Daily Rate)', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Not Canceled', 'Canceled'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('adr_vs_cancellation.png')
plt.close()

# 3. Is Repeated Guest vs. Cancellation
# 'Is Repeated Guest' tells us if the customer has stayed at the hotel before (1 for yes, 0 for no).
# A 'countplot' shows us how many bookings fall into each category (repeated/not repeated)
# and then further breaks it down by whether they canceled.
plt.figure(figsize=(10, 6))
sns.countplot(x='is_repeated_guest', hue='is_canceled', data=df, palette='viridis') # 'hue' splits bars by 'is_canceled'.
plt.title('Is Repeated Guest vs. Cancellation (0 = Not Canceled, 1 = Canceled)', fontsize=14)
plt.xlabel('Is Repeated Guest', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Not Repeated', 'Repeated'])
plt.legend(title='Is Canceled', labels=['Not Canceled', 'Canceled'], loc='upper right') # Adds a legend to explain colors.
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('is_repeated_guest_vs_cancellation.png')
plt.close()

# 4. Previous Cancellations vs. Cancellation
# 'Previous Cancellations' (P-C) is the number of times this customer canceled previous bookings.
# Another box plot to see if past cancellation behavior predicts future cancellations.
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_canceled', y='previous_cancellations', data=df)
plt.title('Previous Cancellations vs. Cancellation (0 = Not Canceled, 1 = Canceled)', fontsize=14)
plt.xlabel('Is Canceled', fontsize=12)
plt.ylabel('Previous Cancellations', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Not Canceled', 'Canceled'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('previous_cancellations_vs_cancellation.png')
plt.close()

# --- 3. K-Nearest Neighbors (KNN) Algorithm Implementation (Making Our Prediction Model) ---

print("\n--- KNN Model Preparation and Training ---")

# Define features (X) and target (y)
# 'Features' (X) are the pieces of information we use to make a prediction (e.g., lead time, price).
# 'Target' (y) is what we want to predict (e.g., 'is_canceled').
features = [
    'numberofadults', # Use cleaned names
    'numberofchildren', # Use cleaned names
    'numberofweekendnights', # Use cleaned names
    'numberofweeknights', # Use cleaned names
    'carparkingspace', # Use cleaned names
    'leadtime', # Use cleaned names
    'is_repeated_guest',
    'previous_cancellations',
    'pnotc', # This is 'P-not-C' from the original data: number of previous bookings NOT canceled.
    'adr',
    'specialrequests' # Use cleaned names
]

X = df[features] # X contains all the columns we'll use to predict.
y = df['is_canceled'] # y contains the column we want to predict (whether it's canceled).

# Handle potential weird values in 'adr' (Average Daily Rate).
# Sometimes, data can have 'infinity' or 'not a number' values, which break models.
# We replace them with the average 'adr' value to make the data clean.
# FIX: Use .loc to avoid SettingWithCopyWarning
X.loc[:, 'adr'] = X['adr'].replace([np.inf, -np.inf], np.nan) # Replace infinity with 'Not a Number'.
X.loc[:, 'adr'] = X['adr'].fillna(X['adr'].mean()) # Fill 'Not a Number' with the average 'adr'.

# Split the data into training and testing sets.
# We split our data so we can train our model on one part (training set)
# and then test how well it performs on data it has never seen before (testing set).
# test_size=0.3 means 30% of the data is for testing, 70% for training.
# random_state=42 ensures that every time we run this code, we get the same split (for consistent results).
# stratify=y ensures that the proportion of canceled/not canceled bookings is roughly the same in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}") # Shows how many rows and columns are in our training data.
print(f"Testing set shape: {X_test.shape}") # Shows how many rows and columns are in our testing data.

# Feature Scaling: Standardize numerical features.
# KNN works by calculating distances between data points. If one feature (like 'lead_time') has very large
# numbers and another (like 'car_parking_space') has small numbers, the larger numbers will unfairly
# dominate the distance calculation. Scaling makes all features contribute equally.
scaler = StandardScaler() # Creates a 'scaler' tool.
X_train_scaled = scaler.fit_transform(X_train) # The scaler 'learns' from our training data and then scales it.
X_test_scaled = scaler.transform(X_test) # We use the same scaler to transform our test data.

# Initialize the KNN Classifier model.
# This is where we set up our K-Nearest Neighbors algorithm.
# n_neighbors=5: This is the 'K' in KNN. It means the model will look at the 5 closest data points
#                to make a prediction for a new data point.
# metric='minkowski', p=2: This specifies how distances are calculated. 'Minkowski' with p=2 is
#                          the same as 'Euclidean distance', which is the straight-line distance.
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# Train the KNN model using the scaled training data.
# This is where the KNN model 'learns' from our data. It essentially memorizes the positions of all
# the training data points along with their 'is_canceled' status.
print("\nTraining KNN model...")
knn.fit(X_train_scaled, y_train) # We tell the model to learn from our scaled features (X_train_scaled)
                                 # and their corresponding cancellation status (y_train).
print("KNN model training complete.")

# Make predictions on the scaled test data.
# Now that our model is trained, we give it the scaled test data (which it hasn't seen before)
# and ask it to predict whether each booking will be canceled or not.
y_pred = knn.predict(X_test_scaled) # 'y_pred' will contain the model's predictions (0 or 1).

# --- 4. Model Evaluation (How Well Did Our Model Do?) ---

print("\n--- KNN Model Evaluation ---")

# Calculate evaluation metrics.
# These numbers tell us how good our model's predictions were compared to the actual outcomes.
accuracy = accuracy_score(y_test, y_pred) # Overall accuracy: (correct predictions) / (total predictions)
precision = precision_score(y_test, y_pred) # How many of the predicted 'canceled' bookings were actually canceled.
recall = recall_score(y_test, y_pred) # How many of the actual 'canceled' bookings did our model correctly find.
f1 = f1_score(y_test, y_pred) # A balance between precision and recall.
conf_matrix = confusion_matrix(y_test, y_pred) # A table that shows correct and incorrect predictions.

print(f"Accuracy: {accuracy:.4f}") # Prints accuracy, rounded to 4 decimal places.
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix) # Shows the raw numbers of correct/incorrect predictions.

# Visualize the Confusion Matrix.
# This plot makes the confusion matrix easier to understand.
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Not Canceled', 'Predicted Canceled'], # Labels for the predicted outcomes.
            yticklabels=['Actual Not Canceled', 'Actual Canceled']) # Labels for the actual outcomes.
plt.title('Confusion Matrix for KNN Model', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('knn_confusion_matrix.png') # Saves the confusion matrix plot.
plt.close()

print("\nEDA plots and KNN Confusion Matrix saved as PNG files.")
print("The KNN model has been trained and evaluated.")
