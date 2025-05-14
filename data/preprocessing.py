import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.logger import get_logger
from utils.helpers import check_memory_usage

logger = get_logger(__name__)


def load_csv_data(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load data from a CSV file with proper error handling.

    Args:
        filepath: Path to the CSV file.
        encoding: File encoding.

    Returns:
        DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(filepath, encoding=encoding)
        logger.info(f"Successfully loaded data from {filepath} with shape {df.shape}")
        logger.debug(f"Memory usage: {check_memory_usage(df)}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise


def sample_data(df: pd.DataFrame, sample_size: float, random_state: int = 42) -> pd.DataFrame:
    """
    Sample a DataFrame to reduce its size.

    Args:
        df: DataFrame to sample.
        sample_size: Proportion of data to keep (0.0 to 1.0).
        random_state: Random seed for reproducibility.

    Returns:
        Sampled DataFrame.
    """
    if sample_size < 1.0:
        sampled_df = df.sample(frac=sample_size, random_state=random_state)
        logger.info(f"Sampled data from {df.shape} to {sampled_df.shape} ({sample_size:.1%})")
        return sampled_df
    return df


def clean_and_validate_data(df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame:
    """
    Clean and validate the DataFrame.

    Args:
        df: DataFrame to clean.
        required_columns: List of columns that must be present.

    Returns:
        Cleaned DataFrame.
    """
    # Validate required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()

    # Remove duplicate rows
    original_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    if len(cleaned_df) < original_rows:
        logger.info(f"Removed {original_rows - len(cleaned_df)} duplicate rows")

    # Check for missing values
    missing_counts = cleaned_df.isnull().sum()
    missing_columns = missing_counts[missing_counts > 0]
    if not missing_columns.empty:
        logger.warning(f"Missing values detected in columns: {missing_columns.to_dict()}")

    return cleaned_df


def preprocess_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess customer data.

    Args:
        df: Customer data DataFrame.

    Returns:
        Preprocessed customer DataFrame.
    """
    logger.info("Preprocessing customer data")

    # Make a copy to avoid modifying the original
    customers_df = df.copy()

    # Required columns for customer data
    required_cols = ['customer_key']
    customers_df = clean_and_validate_data(customers_df, required_cols)

    # Convert gender to numeric if present
    if 'gender' in customers_df.columns:
        if customers_df['gender'].dtype == 'object':
            gender_mapping = {'Male': 0, 'Female': 1}
            customers_df['gender_numeric'] = customers_df['gender'].map(gender_mapping)
            logger.debug("Converted gender to numeric")

    # Convert marital status to numeric if present
    if 'marital_status' in customers_df.columns:
        if customers_df['marital_status'].dtype == 'object':
            marital_mapping = {'Single': 0, 'Married': 1}
            customers_df['marital_status_numeric'] = customers_df['marital_status'].map(marital_mapping)
            logger.debug("Converted marital status to numeric")

    # Handle birth date if present
    if 'birth_date' in customers_df.columns:
        try:
            customers_df['birth_date'] = pd.to_datetime(customers_df['birth_date'])
            current_year = pd.Timestamp.now().year
            customers_df['age'] = current_year - customers_df['birth_date'].dt.year
            logger.debug("Calculated age from birth date")
        except Exception as e:
            logger.warning(f"Could not process birth date: {str(e)}")

    # Handle country data with one-hot encoding if needed
    if 'country' in customers_df.columns:
        if customers_df['country'].dtype == 'object':
            # Only one-hot encode if there aren't too many countries
            unique_countries = customers_df['country'].nunique()
            if unique_countries <= 20:  # Arbitrary threshold
                country_dummies = pd.get_dummies(
                    customers_df['country'],
                    prefix='country',
                    dummy_na=False
                )
                customers_df = pd.concat([customers_df, country_dummies], axis=1)
                logger.debug(f"One-hot encoded {unique_countries} countries")

    # Process age group if present
    if 'age_group' in customers_df.columns:
        if customers_df['age_group'].dtype == 'object':
            # Extract lower bound from age range as numeric
            try:
                # Example: '40-49' -> 40
                customers_df['age_group_lower'] = customers_df['age_group'].str.extract(r'(\d+)').astype(float)
                logger.debug("Extracted numeric age group values")
            except Exception as e:
                logger.warning(f"Could not process age group: {str(e)}")

    logger.info(f"Customer preprocessing complete. Final shape: {customers_df.shape}")
    return customers_df


def preprocess_product_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess product data.

    Args:
        df: Product data DataFrame.

    Returns:
        Preprocessed product DataFrame.
    """
    logger.info("Preprocessing product data")

    # Make a copy to avoid modifying the original
    products_df = df.copy()

    # Required columns for product data
    required_cols = ['product_key']
    products_df = clean_and_validate_data(products_df, required_cols)

    # Create a standardized product description from available fields
    text_fields = ['product_name', 'category', 'sub_category']
    available_fields = [field for field in text_fields if field in products_df.columns]

    if available_fields:
        # Combine all available text fields
        products_df['product_description'] = products_df[available_fields].fillna('').agg(' '.join, axis=1)
        logger.debug(f"Created product description from fields: {available_fields}")

    # Convert categorical features to one-hot encoding
    categorical_fields = ['category', 'sub_category', 'maintenance', 'product_ine']
    available_categorical = [field for field in categorical_fields if field in products_df.columns]

    for field in available_categorical:
        if products_df[field].dtype == 'object':
            field_dummies = pd.get_dummies(
                products_df[field],
                prefix=field,
                dummy_na=False
            )
            products_df = pd.concat([products_df, field_dummies], axis=1)
            logger.debug(f"One-hot encoded {field} with {field_dummies.shape[1]} categories")

    # Handle numerical features
    if 'cost' in products_df.columns:
        # Fill missing costs with median
        if products_df['cost'].isnull().sum() > 0:
            median_cost = products_df['cost'].median()
            products_df['cost'] = products_df['cost'].fillna(median_cost)
            logger.debug(f"Filled missing cost values with median: {median_cost}")

    logger.info(f"Product preprocessing complete. Final shape: {products_df.shape}")
    return products_df


def preprocess_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess sales transaction data.

    Args:
        df: Sales data DataFrame.

    Returns:
        Preprocessed sales DataFrame.
    """
    logger.info("Preprocessing sales data")

    # Make a copy to avoid modifying the original
    sales_df = df.copy()

    # Required columns for sales data
    required_cols = ['customer_key', 'product_key']
    sales_df = clean_and_validate_data(sales_df, required_cols)

    # Convert dates if present
    date_columns = ['order_date', 'shipping_date', 'due_date']
    for col in date_columns:
        if col in sales_df.columns:
            try:
                sales_df[col] = pd.to_datetime(sales_df[col])
                logger.debug(f"Converted {col} to datetime")
            except Exception as e:
                logger.warning(f"Could not convert {col} to datetime: {str(e)}")

    # Create a rating column if not present
    if 'rating' not in sales_df.columns:
        # Use sales or quantity as implicit rating
        if 'sales' in sales_df.columns:
            # Normalize sales to a 1-5 scale
            sales_min = sales_df['sales'].min()
            sales_max = sales_df['sales'].max()
            if sales_max > sales_min:
                sales_df['rating'] = 1 + 4 * (sales_df['sales'] - sales_min) / (sales_max - sales_min)
                logger.debug("Created rating column from sales values")
            else:
                sales_df['rating'] = 1.0  # Default if all values are the same
        elif 'quantity' in sales_df.columns:
            # Use quantity as an implicit signal of preference
            sales_df['rating'] = sales_df['quantity'].clip(1, 5)  # Clip to 1-5 range
            logger.debug("Created rating column from quantity values")
        else:
            # If no value is available, use a default rating for all
            sales_df['rating'] = 1.0
            logger.debug("Created default rating column with value 1.0")

    # Drop duplicate user-item interactions, keeping the most recent or highest rated
    if 'order_date' in sales_df.columns and pd.api.types.is_datetime64_dtype(sales_df['order_date']):
        # Sort by date (descending) and keep the most recent interaction
        sales_df = sales_df.sort_values('order_date', ascending=False)
        sales_df = sales_df.drop_duplicates(subset=['customer_key', 'product_key'], keep='first')
        logger.debug("Removed duplicate interactions, keeping most recent")
    elif 'rating' in sales_df.columns:
        # Sort by rating (descending) and keep the highest rated interaction
        sales_df = sales_df.sort_values('rating', ascending=False)
        sales_df = sales_df.drop_duplicates(subset=['customer_key', 'product_key'], keep='first')
        logger.debug("Removed duplicate interactions, keeping highest rated")

    logger.info(f"Sales preprocessing complete. Final shape: {sales_df.shape}")
    return sales_df


def create_tfidf_features(
        df: pd.DataFrame,
        text_column: str,
        max_features: Optional[int] = 100,
        min_df: float = 0.01,
        max_df: float = 0.95
) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Create TF-IDF features from a text column.

    Args:
        df: DataFrame containing the text data.
        text_column: Column name with text to vectorize.
        max_features: Maximum number of features to extract.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency.

    Returns:
        DataFrame with TF-IDF features and the fitted vectorizer.
    """
    logger.info(f"Creating TF-IDF features from column: {text_column}")

    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words='english'
    )

    # Ensure text data is string type
    text_data = df[text_column].fillna('').astype(str)

    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(text_data)

    # Convert to DataFrame
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=df.index,
        columns=[f"tfidf_{text_column}_{name}" for name in feature_names]
    )

    logger.info(f"Created {len(feature_names)} TF-IDF features")

    return tfidf_df, vectorizer