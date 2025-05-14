import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.sparse import csr_matrix

from utils.logger import get_logger
from data.preprocessing import (
    load_csv_data,
    sample_data,
    preprocess_customer_data,
    preprocess_product_data,
    preprocess_sales_data
)

logger = get_logger(__name__)


def get_nested_dict_value(d: Dict[str, Any], key_path: str, default=None):
    """
    Get a value from a nested dictionary using dot notation.

    Args:
        d: Dictionary to search
        key_path: Path to the value using dot notation (e.g., 'data.paths.customers')
        default: Default value if key not found

    Returns:
        Value at the specified path or default if not found
    """
    keys = key_path.split('.')
    result = d

    try:
        for k in keys:
            result = result[k]
        return result
    except (KeyError, TypeError):
        return default


class RetailDataset:
    """
    Dataset class for retail recommendation data.

    This class handles loading, preprocessing, and managing the retail dataset
    with customer demographics, product information, and purchase histories.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RetailDataset.

        Args:
            config: Configuration dictionary containing data paths and settings.
        """
        self.config = config

        # Get data directory
        self.data_dir = get_nested_dict_value(config, 'data.dataset_dir')
        if not self.data_dir:
            # Try direct access for backwards compatibility
            self.data_dir = config.get('data_dir') or './data/datasets'
            logger.warning(f"data.dataset_dir not found, using: {self.data_dir}")

        # Get file paths
        data_paths = get_nested_dict_value(config, 'data.paths', {})

        # Set paths with fallbacks for each file
        self.customers_path = os.path.join(self.data_dir,
                                           get_nested_dict_value(data_paths, 'customers',
                                                                 'g_dim_customers.csv'))
        self.products_path = os.path.join(self.data_dir,
                                          get_nested_dict_value(data_paths, 'products',
                                                                'g_dim_products.csv'))
        self.sales_path = os.path.join(self.data_dir,
                                       get_nested_dict_value(data_paths, 'sales',
                                                             'g_fact_sales.csv'))
        self.customer_report_path = os.path.join(self.data_dir,
                                                 get_nested_dict_value(data_paths, 'customer_report',
                                                                       'Customer_report_cleaned_data.csv'))
        self.product_report_path = os.path.join(self.data_dir,
                                                get_nested_dict_value(data_paths, 'product_report',
                                                                      'Product_report_cleaned_data.csv'))

        # Get other settings
        self.encoding = get_nested_dict_value(config, 'data.encoding', 'utf-8')
        self.sample_size = get_nested_dict_value(config, 'data.sample_size', 1.0)
        self.random_seed = get_nested_dict_value(config, 'data.random_seed', 42)

        # Data containers
        self.customers = None
        self.products = None
        self.sales = None
        self.customer_report = None
        self.product_report = None

        # Derived data
        self.user_item_matrix = None
        self.user_item_csr_matrix = None
        self.user_mapping = None
        self.item_mapping = None

        logger.info(f"RetailDataset initialized with sample_size={self.sample_size}")

    # All other methods remain the same
    def load_data(self) -> None:
        """
        Load all data sources and apply initial preprocessing.
        """
        logger.info("Loading retail dataset data...")

        # Load customers data
        if os.path.exists(self.customers_path):
            logger.info(f"Loading customers from {self.customers_path}")
            self.customers = load_csv_data(self.customers_path, self.encoding)
            self.customers = sample_data(self.customers, self.sample_size, self.random_seed)
            self.customers = preprocess_customer_data(self.customers)
        else:
            logger.warning(f"Customers file not found: {self.customers_path}")

        # Load products data
        if os.path.exists(self.products_path):
            logger.info(f"Loading products from {self.products_path}")
            self.products = load_csv_data(self.products_path, self.encoding)
            # No sampling for products as we need the full catalog
            self.products = preprocess_product_data(self.products)
        else:
            logger.warning(f"Products file not found: {self.products_path}")

        # Load sales data
        if os.path.exists(self.sales_path):
            logger.info(f"Loading sales from {self.sales_path}")
            self.sales = load_csv_data(self.sales_path, self.encoding)

            # If we've sampled customers, filter sales to only include those customers
            if self.customers is not None and self.sample_size < 1.0:
                original_sales_count = len(self.sales)
                self.sales = self.sales[self.sales['customer_key'].isin(self.customers['customer_key'])]
                logger.info(
                    f"Filtered sales from {original_sales_count} to {len(self.sales)} rows based on sampled customers")

            self.sales = preprocess_sales_data(self.sales)
        else:
            logger.warning(f"Sales file not found: {self.sales_path}")

        # Load customer report data (if available)
        if os.path.exists(self.customer_report_path):
            logger.info(f"Loading customer report from {self.customer_report_path}")
            self.customer_report = load_csv_data(self.customer_report_path, self.encoding)

            # Sample and filter if necessary
            if self.sample_size < 1.0 and 'customer_key' in self.customer_report.columns:
                if self.customers is not None:
                    self.customer_report = self.customer_report[
                        self.customer_report['customer_key'].isin(self.customers['customer_key'])
                    ]
                    logger.info(
                        f"Filtered customer report to {len(self.customer_report)} rows based on sampled customers")

        # Load product report data (if available)
        if os.path.exists(self.product_report_path):
            logger.info(f"Loading product report from {self.product_report_path}")
            self.product_report = load_csv_data(self.product_report_path, self.encoding)

        logger.info("All data loaded successfully")

    def create_interaction_matrix(self) -> pd.DataFrame:
        """
        Create a user-item interaction matrix from the sales data.

        Returns:
            User-item interaction matrix.
        """
        if self.sales is None:
            raise ValueError("Sales data not loaded. Call load_data() first.")

        logger.info("Creating user-item interaction matrix")

        # Create a pivot table with customers as rows and products as columns
        if 'rating' in self.sales.columns:
            interaction_value = 'rating'
        elif 'sales' in self.sales.columns:
            interaction_value = 'sales'
        elif 'quantity' in self.sales.columns:
            interaction_value = 'quantity'
        else:
            # If no suitable values, use a binary indicator
            self.sales['interaction'] = 1
            interaction_value = 'interaction'

        # Create the interaction matrix
        self.user_item_matrix = self.sales.pivot_table(
            index='customer_key',
            columns='product_key',
            values=interaction_value,
            aggfunc='mean',
            fill_value=0
        )

        # Create mappings between original IDs and matrix indices
        self.user_mapping = {
            user_id: i for i, user_id in enumerate(self.user_item_matrix.index)
        }
        self.item_mapping = {
            item_id: i for i, item_id in enumerate(self.user_item_matrix.columns)
        }

        # Create a sparse matrix representation
        self.user_item_csr_matrix = csr_matrix(self.user_item_matrix.values)

        logger.info(f"Created interaction matrix with shape {self.user_item_matrix.shape}")
        sparsity = 100.0 * (1.0 - self.user_item_csr_matrix.nnz /
                            (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]))
        logger.info(f"Matrix sparsity: {sparsity:.2f}%")

        return self.user_item_matrix

    def get_user_features(self) -> pd.DataFrame:
        """
        Get user features for content-based and hybrid systems.

        Returns:
            DataFrame with user features.
        """
        if self.customers is None:
            raise ValueError("Customer data not loaded. Call load_data() first.")

        # Start with demographic features from customers
        user_features = self.customers.set_index('customer_key')

        # Add features from customer report if available
        if self.customer_report is not None and 'customer_key' in self.customer_report.columns:
            report_features = self.customer_report.set_index('customer_key')

            # Select only numeric columns for simplicity
            numeric_cols = report_features.select_dtypes(include=np.number).columns

            # Check for overlapping columns to avoid duplicates
            non_overlapping = [col for col in numeric_cols if col not in user_features.columns]

            if non_overlapping:
                user_features = user_features.join(report_features[non_overlapping], how='left')
                logger.info(f"Added {len(non_overlapping)} features from customer report")

        logger.info(f"Generated user features with shape {user_features.shape}")
        return user_features

    def get_item_features(self) -> pd.DataFrame:
        """
        Get item features for content-based and hybrid systems.

        Returns:
            DataFrame with item features.
        """
        if self.products is None:
            raise ValueError("Product data not loaded. Call load_data() first.")

        # Start with features from products
        item_features = self.products.set_index('product_key')

        # Add features from product report if available
        if self.product_report is not None and 'product_key' in self.product_report.columns:
            report_features = self.product_report.set_index('product_key')

            # Select only numeric columns for simplicity
            numeric_cols = report_features.select_dtypes(include=np.number).columns

            # Check for overlapping columns to avoid duplicates
            non_overlapping = [col for col in numeric_cols if col not in item_features.columns]

            if non_overlapping:
                item_features = item_features.join(report_features[non_overlapping], how='left')
                logger.info(f"Added {len(non_overlapping)} features from product report")

        logger.info(f"Generated item features with shape {item_features.shape}")
        return item_features

    def get_interaction_data(self) -> pd.DataFrame:
        """
        Get processed interaction data.

        Returns:
            DataFrame with user-item interactions.
        """
        if self.sales is None:
            raise ValueError("Sales data not loaded. Call load_data() first.")

        # Select relevant columns
        interaction_data = self.sales[['customer_key', 'product_key']].copy()

        # Add rating column if available
        if 'rating' in self.sales.columns:
            interaction_data['rating'] = self.sales['rating']
        elif 'sales' in self.sales.columns:
            interaction_data['rating'] = self.sales['sales']
        elif 'quantity' in self.sales.columns:
            interaction_data['rating'] = self.sales['quantity']
        else:
            interaction_data['rating'] = 1.0

        # Add timestamp if available
        if 'order_date' in self.sales.columns:
            interaction_data['timestamp'] = self.sales['order_date']

        logger.info(f"Prepared interaction data with shape {interaction_data.shape}")
        return interaction_data

    def get_customer_segments(self) -> Dict[str, List[int]]:
        """
        Get customer segments if available.

        Returns:
            Dictionary with segment names as keys and lists of customer IDs as values.
        """
        segments = {}

        if self.customer_report is not None and 'cust_segmentation' in self.customer_report.columns:
            segment_column = 'cust_segmentation'
            for segment in self.customer_report[segment_column].unique():
                if segment and not pd.isna(segment):
                    segment_customers = self.customer_report[
                        self.customer_report[segment_column] == segment
                        ]['customer_key'].tolist()
                    segments[segment] = segment_customers
                    logger.info(f"Found segment '{segment}' with {len(segment_customers)} customers")

        return segments

    def get_product_segments(self) -> Dict[str, List[int]]:
        """
        Get product segments if available.

        Returns:
            Dictionary with segment names as keys and lists of product IDs as values.
        """
        segments = {}

        if self.product_report is not None and 'product_segment' in self.product_report.columns:
            segment_column = 'product_segment'
            for segment in self.product_report[segment_column].unique():
                if segment and not pd.isna(segment):
                    segment_products = self.product_report[
                        self.product_report[segment_column] == segment
                        ]['product_key'].tolist()
                    segments[segment] = segment_products
                    logger.info(f"Found segment '{segment}' with {len(segment_products)} products")

        return segments

    def get_user_id_mapping(self) -> Dict[int, int]:
        """
        Get mapping from original user IDs to matrix indices.

        Returns:
            Dictionary mapping user IDs to indices.
        """
        if self.user_mapping is None:
            self.create_interaction_matrix()
        return self.user_mapping

    def get_item_id_mapping(self) -> Dict[int, int]:
        """
        Get mapping from original item IDs to matrix indices.

        Returns:
            Dictionary mapping item IDs to indices.
        """
        if self.item_mapping is None:
            self.create_interaction_matrix()
        return self.item_mapping

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics.
        """
        stats = {
            'sample_size': self.sample_size,
        }

        if self.customers is not None:
            stats['num_customers'] = len(self.customers)

        if self.products is not None:
            stats['num_products'] = len(self.products)

            if 'category' in self.products.columns:
                stats['product_categories'] = self.products['category'].value_counts().to_dict()

        if self.sales is not None:
            stats['num_interactions'] = len(self.sales)

            if self.customers is not None and self.products is not None:
                stats['interactions_density'] = len(self.sales) / (len(self.customers) * len(self.products))

        if self.user_item_matrix is not None:
            stats['interaction_matrix_shape'] = self.user_item_matrix.shape
            stats['interaction_matrix_sparsity'] = 1.0 - (self.user_item_csr_matrix.nnz /
                                                          (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[
                                                              1]))

        logger.info(f"Dataset statistics: {stats}")
        return stats