import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide", initial_sidebar_state="expanded")

# Set the background color to white using custom CSS
st.markdown("""
    <style>
    .reportview-container {
        background: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load your pre-trained model
model = joblib.load('RFClassifier.pkl')

st.title('Customer Churn Prediction and Analysis')

# Function to get user input
def get_user_input():
    st.sidebar.header('Input Parameters')
    subscription_type = st.sidebar.selectbox('Subscription Type', ('Basic', 'Standard', 'Premium'))
    age = st.sidebar.slider('Age', 18, 70, 30)
    monthly_revenue = st.sidebar.slider('Monthly Revenue', 10, 100, 50)
    device = st.sidebar.selectbox('Device', ('Smartphone', 'Tablet', 'Smart TV', 'Laptop'))
    country = st.sidebar.selectbox('Country', ('United States', 'Canada', 'United Kingdom'))
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    plan_duration = st.sidebar.selectbox('Plan Duration', ('1 Month', '3 Months', '6 Months', '12 Months'))

    # Create a dictionary for the input
    user_data = {'Subscription Type': subscription_type,
                 'Age': age,
                 'Monthly Revenue': monthly_revenue,
                 'Device': device,
                 'Country': country,
                 'Gender': gender,
                 'Plan Duration': plan_duration}

    features = pd.DataFrame(user_data, index=[0])
    return features

# Sidebar for user input
input_df = get_user_input()

# Display the user input
st.subheader('User Input Parameters')
st.write(input_df)

# Prepare the input for prediction
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display the prediction
st.subheader('Prediction')
st.write('🎯 Churn' if prediction[0] == 1 else '🛡️ No Churn')

st.subheader('Prediction Probability')
st.write(f"📊 Probability of Churn: {prediction_proba[0][1]:.2f}")

# Load and preprocess data for visualization
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv("Netflix Userbase.csv")
    data['Last Payment Date'] = pd.to_datetime(data['Last Payment Date'], dayfirst=True)
    data['Join Date'] = pd.to_datetime(data['Join Date'], dayfirst=True)
    end_date = datetime.strptime('07-12-2023', '%d-%m-%Y')
    data['Days Since Last Payment'] = (end_date - data['Last Payment Date']).dt.days
    data['Churn'] = (data['Days Since Last Payment'] > 355).astype(int)  # Example threshold
    return data

data = load_and_preprocess_data()

# Define thresholds and calculate churn metrics
def calculate_churn_metrics(data, thresholds):
    churn_rates = []
    for threshold in thresholds:
        data[f'Churn_{threshold}'] = (data['Days Since Last Payment'] > threshold).astype(int)
        churn_rate = data[f'Churn_{threshold}'].mean()
        churn_rates.append(churn_rate)
    return churn_rates

thresholds = [30, 60, 90, 120, 180]
churn_rates = calculate_churn_metrics(data, thresholds)

# Button for churn rate visualization
st.subheader('Churn Rate Visualization')

if st.button('Show Churn Rate Visualization'):
    st.subheader('Churn Rate vs. Threshold')
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, churn_rates, marker='o', linestyle='-', color='b')
    plt.xlabel('Threshold (Days)')
    plt.ylabel('Churn Rate')
    plt.grid(True)
    st.pyplot(plt)

# Additional Visualizations

# Distribution of Age and Monthly Revenue
st.subheader('Distribution of Age and Monthly Revenue')

col1, col2 = st.columns(2)

with col1:
    if st.button('Show Age Distribution'):
        st.subheader('Age Distribution')
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Age'], bins=20, kde=True, color='skyblue')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution')
        st.pyplot(plt)

with col2:
    if st.button('Show Monthly Revenue Distribution'):
        st.subheader('Monthly Revenue Distribution')
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Monthly Revenue'], bins=20, kde=True, color='salmon')
        plt.xlabel('Monthly Revenue')
        plt.ylabel('Frequency')
        plt.title('Monthly Revenue Distribution')
        st.pyplot(plt)

# Churn Count by Device and Country
st.subheader('Churn Count by Device and Country')

col1, col2 = st.columns(2)

with col1:
    if st.button('Show Churn Count by Device'):
        st.subheader('Churn Count by Device')
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Device', hue='Churn', data=data, palette='viridis')
        plt.xlabel('Device')
        plt.ylabel('Count')
        plt.title('Churn Count by Device')
        plt.xticks(rotation=45)
        st.pyplot(plt)

with col2:
    if st.button('Show Churn Count by Country'):
        st.subheader('Churn Count by Country')
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Country', hue='Churn', data=data, palette='viridis')
        plt.xlabel('Country')
        plt.ylabel('Count')
        plt.title('Churn Count by Country')
        plt.xticks(rotation=45)
        st.pyplot(plt)

# Correlation Heatmap
st.subheader('Correlation Heatmap')

if st.button('Show Correlation Heatmap'):
    st.subheader('Correlation Heatmap of Numerical Features')
    plt.figure(figsize=(12, 8))
    corr = data[['Age', 'Monthly Revenue', 'Days Since Last Payment', 'Churn']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

# Clustering Visualization
st.subheader('Clustering Visualization')

def perform_clustering(data, n_clusters=3):
    features = data[['Age', 'Monthly Revenue']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['Cluster'] = kmeans.fit_predict(scaled_features)
    return data, kmeans

if st.button('Show Clustering Visualization'):
    st.subheader('Clustering Based on Age and Monthly Revenue')
    n_clusters = st.slider('Number of Clusters', min_value=2, max_value=10, value=3)
    clustered_data, kmeans_model = perform_clustering(data, n_clusters)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Age', y='Monthly Revenue', hue='Cluster', data=clustered_data, palette='viridis', marker='o')
    plt.xlabel('Age')
    plt.ylabel('Monthly Revenue')
    plt.title(f'Clustering of Users (n_clusters={n_clusters})')
    plt.legend(title='Cluster')
    st.pyplot(plt)

# Visualizations for various attributes
def plot_churn_by_attribute(data, attribute):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=data[attribute].value_counts().index, y=data[attribute].value_counts().values, palette='viridis')
    plt.xlabel(attribute)
    plt.ylabel('Count')
    plt.title(f'Churn Count by {attribute}')
    plt.grid(True)
    st.pyplot(plt)

def plot_churn_rate_by_attribute(data, attribute):
    churn_rate_by_attr = data.groupby(attribute)['Churn'].mean().reset_index()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=attribute, y='Churn', data=churn_rate_by_attr, palette='viridis')
    plt.xlabel(attribute)
    plt.ylabel('Churn Rate')
    plt.title(f'Churn Rate by {attribute}')
    plt.grid(True)
    st.pyplot(plt)

# Show visualizations for different attributes
attributes = ['Age', 'Monthly Revenue', 'Device', 'Subscription Type']
selected_attr = st.selectbox('Select Attribute to Visualize', attributes)

# Assign a unique key to each button
if st.button(f'Show Churn Count by {selected_attr}', key=f'churn_count_{selected_attr}'):
    plot_churn_by_attribute(data, selected_attr)

if st.button(f'Show Churn Rate by {selected_attr}', key=f'churn_rate_{selected_attr}'):
    plot_churn_rate_by_attribute(data, selected_attr)


# Additional visualizations for churn analysis
def plot_churn_rate_distribution(data):
    plt.figure(figsize=(12, 8))
    sns.histplot(data['Days Since Last Payment'], bins=30, kde=True, color='b')
    plt.xlabel('Days Since Last Payment')
    plt.ylabel('Frequency')
    plt.title('Distribution of Days Since Last Payment')
    plt.grid(True)
    st.pyplot(plt)

def plot_churn_over_time(data):
    data['Month'] = data['Last Payment Date'].dt.to_period('M')
    monthly_churn = data.groupby('Month')['Churn'].mean().reset_index()
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=monthly_churn['Month'].astype(str), y=monthly_churn['Churn'], marker='o', color='g')
    plt.xlabel('Month')
    plt.ylabel('Churn Rate')
    plt.title('Churn Rate Over Time')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

def plot_churn_heatmap(data):
    churn_pivot = data.pivot_table(index='Device', columns='Subscription Type', values='Churn', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(churn_pivot, cmap='YlGnBu', annot=True, fmt='.2f')
    plt.xlabel('Subscription Type')
    plt.ylabel('Device')
    plt.title('Churn Rate Heatmap by Device and Subscription Type')
    plt.grid(True)
    st.pyplot(plt)

# Show additional visualizations
if st.button('Show Churn Rate Distribution'):
    plot_churn_rate_distribution(data)

if st.button('Show Churn Over Time'):
    plot_churn_over_time(data)

if st.button('Show Churn Heatmap'):
    plot_churn_heatmap(data)


# Footer
st.sidebar.markdown("""
---
### About
This dashboard provides insights into customer churn predictions using a pre-trained machine learning model.
Developed by [Your Name].
""")
