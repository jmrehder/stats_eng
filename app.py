import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical packages
import scipy.stats as stats
import statsmodels.api as sm

# For the PDF report
from fpdf import FPDF
from io import BytesIO

#########################################
# Main Function & Sidebar (7 Options)  #
#########################################
def main():
    st.title("Statistical Analysis by JM")
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Select a View",
        [
            "Homepage",
            "Load Seaborn Dataset",
            "File Upload",
            "Data Exploration",
            "Data Cleaning",
            "Descriptive Statistics",
            "Hypothesis Manager",
            "Statistical Analyses (t-Test, Correlation, Chi², Regression)",
            "Predictions",
            "PDF Report"
        ]
    )
    
    if app_mode == "Homepage":
        homepage()
    if app_mode == "Load Seaborn Dataset":
        seaborn_datasets()
    elif app_mode == "File Upload":
        file_uploader()
    elif app_mode == "Data Exploration":
        data_exploration()
    elif app_mode == "Data Cleaning":
        data_cleaning()
    elif app_mode == "Descriptive Statistics":
        descriptive_statistics()
    elif app_mode == "Hypothesis Manager":
        hypothesis_manager()
    elif app_mode == "Statistical Analyses (t-Test, Correlation, Chi², Regression)":
        advanced_analyses()
    elif app_mode == "Predictions":
        predictions()
    elif app_mode == "PDF Report":
        pdf_report()


#########################################
# Function: Homepage                    #
#########################################
def homepage():
    st.header("Welcome to Statistical Analyses with Streamlit")
    st.write(
        """
        This application offers you an interactive way to perform statistical analyses. 
        Choose a feature from the sidebar to get started.

        **Features:**
        - **Load Seaborn Dataset:** Use predefined datasets from Seaborn for your analyses.
        - **File Upload:** Upload your own CSV or Excel files.
        - **Data Exploration:** Explore your data through visual and statistical summaries.
        - **Data Cleaning:** Remove missing values and clean your dataset.
        - **Descriptive Statistics:** Get basic metrics and visualizations of your data.
        - **Hypothesis Manager:** Create null and alternative hypotheses based on your variables.
        - **Statistical Analyses:** Perform t-tests, correlations, Chi² tests, and regressions.
        - **Predictions:** Use trained regression models to make predictions.
        - **PDF Report:** Export your results in a structured PDF report.

        **How it works:**
        1. Upload your data or choose a predefined dataset.
        2. Select a feature from the sidebar.
        3. Follow the on-screen instructions to conduct your analyses.

        Enjoy using this application!
        """
    )

#########################################
# Function: Load Seaborn Datasets       #
#########################################
def seaborn_datasets():
    st.header("Load Seaborn Datasets")
    st.info(
        """
        Here, you can load a predefined dataset from the Seaborn library.
        Click the button to get a description and load the dataset.
        """
    )
    
    dataset_descriptions = {
        "anscombe": "Four datasets demonstrating how identical statistics can hide different distributions.",
        "attention": "A dataset examining participant attention during different tasks.",
        "car_crashes": "Data on car crashes in various US states.",
        "diamonds": "A dataset with diamond prices and characteristics.",
        "dots": "Movement data of dots on a screen.",
        "exercise": "Data on physical exercises and their health effects.",
        "flights": "Monthly air passenger numbers over several years.",
        "fmri": "fMRI data from participants under different conditions.",
        "gammas": "Data on gamma radiation measurements.",
        "iris": "Measurements of iris flowers (length and width of sepals and petals).",
        "penguins": "Data on various penguin species, including weight and flipper length.",
        "planets": "Discovered exoplanets and their properties.",
        "tips": "Restaurant tipping data.",
        "titanic": "Passenger data from the Titanic, including survival status and classes.",
    }
    
    dataset_names = sns.get_dataset_names()
    
    selected_dataset = st.selectbox("Choose a dataset", dataset_names)
    if st.button("Show Description"):
        description = dataset_descriptions.get(selected_dataset, "No description available.")
        st.write(f"**Description of {selected_dataset}:**")
        st.write(description)
    
    if st.button("Load Dataset"):
        df = sns.load_dataset(selected_dataset)
        st.session_state["df"] = df
        st.success(f"Dataset '{selected_dataset}' loaded successfully!")
        st.write("Preview of the dataset:")
        st.dataframe(df.head())

#########################################
# Function: Load Data                   #
#########################################
@st.cache_data
def load_data(file, file_type):
    """Loads data from a CSV or Excel file into a DataFrame."""
    if file_type == "csv":
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

#########################################
# Function: File Upload                 #
#########################################
def file_uploader():
    st.header("Upload a Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            file_type = "csv"
        else:
            file_type = "excel"
        df = load_data(uploaded_file, file_type)
        st.session_state["df"] = df
        st.success("File uploaded successfully!")
        st.write("Preview of the dataset:")
        st.dataframe(df.head())

#########################################
# Function: Data Exploration            #
#########################################
def data_exploration():
    st.header("Initial Data Exploration")
    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state["df"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Dataset Shape")
        st.write("**Shape:**", df.shape)
    with col2:
        st.markdown("### Column List")
        st.write(df.columns.tolist())
    
    st.markdown("---")
    st.subheader("Data Preview")
    num_rows = st.slider("Number of rows to display", 1, 100, 5)
    st.dataframe(df.head(num_rows))
    
    st.markdown("---")
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    missing_count = missing_values[missing_values > 0]
    if missing_count.empty:
        st.success("No missing values found!")
    else:
        st.write("Missing values in columns:")
        st.dataframe(missing_count.to_frame())

#########################################
# Function: Data Cleaning               #
#########################################
def data_cleaning():
    st.header("Data Cleaning: Remove Missing Values")
    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state["df"]
    st.write("Original number of rows:", df.shape[0])
    
    missing = df.isnull().sum()[df.isnull().sum() > 0]
    if missing.empty:
        st.success("No missing values found!")
        return
    st.write("The following columns contain missing values:")
    st.dataframe(missing.to_frame())
    columns_with_missing = list(missing.index)
    selected_cols = st.multiselect("Select columns to remove rows with missing values:", columns_with_missing)
    if st.button("Remove Missing Values"):
        if not selected_cols:
            st.warning("Please select at least one column.")
        else:
            df_clean = df.dropna(subset=selected_cols)
            st.write("Number of rows after removal:", df_clean.shape[0])
            st.dataframe(df_clean.head())
            st.success("Missing values removed!")
            st.session_state["df"] = df_clean

#########################################
# Function: Descriptive Statistics      #
#########################################
def descriptive_statistics():
    st.header("Descriptive Statistics")
    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state["df"]

    st.subheader("Basic Statistics")
    st.write(df.describe())
    st.info(
        """
        **Interpretation of Metrics:**
        
        - **count**: Number of non-missing observations.
        - **mean**: Average value of the variable.
        - **std**: Standard deviation – measures the spread around the mean.
        - **min** and **max**: Minimum and maximum values, representing the range.
        - **25%, 50%, 75%**: Quantiles, where 50% represents the median.
        
        These metrics provide an initial insight into the distribution of numerical variables.
        """
    )

    st.subheader("Histograms")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        column_to_plot = st.selectbox("Select a column for the histogram", numeric_columns, key="histogram")
        fig, ax = plt.subplots()
        sns.histplot(df[column_to_plot], kde=True, ax=ax)
        ax.set_title(f"Histogram of {column_to_plot}")
        st.pyplot(fig)
    else:
        st.warning("No numerical columns found for histogram.")

    st.subheader("Correlation Matrix")
    if numeric_columns:
        corr = df.corr(numeric_only=True)
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
        ax_corr.set_title("Correlation Matrix")
        st.pyplot(fig_corr)
    else:
        st.warning("No numerical columns found for correlation matrix.")

    st.subheader("Additional Visualizations")
    additional_plot = st.selectbox("Select an additional visualization", ["Boxplot", "Scatterplot", "Violinplot", "Pairplot"])
    if additional_plot == "Boxplot":
        if numeric_columns:
            col = st.selectbox("Select a numerical column for the boxplot", numeric_columns, key="boxplot")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)
        else:
            st.warning("No numerical columns found for boxplot.")
    elif additional_plot == "Scatterplot":
        if len(numeric_columns) >= 2:
            col_x = st.selectbox("Select the X-axis", numeric_columns, key="scatter_x")
            col_y_options = [col for col in numeric_columns if col != col_x]
            col_y = st.selectbox("Select the Y-axis", col_y_options, key="scatter_y")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax)
            ax.set_title(f"Scatterplot: {col_x} vs. {col_y}")
            st.pyplot(fig)
        else:
            st.warning("At least two numerical columns are required for a scatterplot.")
    elif additional_plot == "Violinplot":
        if numeric_columns:
            col = st.selectbox("Select a numerical column for the violin plot", numeric_columns, key="violinplot")
            fig, ax = plt.subplots()
            sns.violinplot(y=df[col], ax=ax)
            ax.set_title(f"Violinplot of {col}")
            st.pyplot(fig)
        else:
            st.warning("No numerical columns found for violin plot.")
    elif additional_plot == "Pairplot":
        if len(numeric_columns) >= 2:
            st.write("Pairplot of numerical columns:")
            pairplot_fig = sns.pairplot(df[numeric_columns].dropna())
            pairplot_fig.savefig("pairplot.png")
            st.image("pairplot.png")
        else:
            st.warning("At least two numerical columns are required for a pairplot.")

#########################################
# Function: Advanced Analyses           #
#########################################
def advanced_analyses():
    st.header("Statistical Analyses: t-Test, Correlation, Chi², Regression")
    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state["df"]

    analysis_type = st.radio(
        "Which analysis would you like to perform?",
        ("t-Test", "Correlation", "Chi²-Test", "Regression")
    )

    # t-Test
    if analysis_type == "t-Test":
        st.subheader("t-Test (independent samples)")
        if st.button("Explanation of t-Test"):
            st.info("""
            **t-Test for Independent Samples**

            The t-test is used to determine whether the means of two groups are significantly different.
            **Examples of Use:**
            - Comparing the average weight of men and women.
            - Comparing test scores between two different classes.

            **Prerequisites:**
            1. **Independence of Groups**: Measurements in one group should not affect the other group.
            2. **Normal Distribution**: The target variable should be approximately normally distributed in both groups.
            3. **Equal Variances**: The variances of the two groups should be similar. If not, an adjusted t-test (Welch test) can be used.
            """)

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
        if not numeric_columns or not categorical_columns:
            st.warning("Your dataset requires at least one numerical and one categorical column.")
            return
        group_col = st.selectbox("Select the group column (categorical)", categorical_columns)
        numeric_col = st.selectbox("Select the numerical variable to compare", numeric_columns)
        groups = df[group_col].unique()
        if len(groups) != 2:
            st.warning("Exactly two groups are required for a t-Test.")
            return
        group1 = df[df[group_col] == groups[0]][numeric_col].dropna()
        group2 = df[df[group_col] == groups[1]][numeric_col].dropna()
        t_stat, p_value = stats.ttest_ind(group1, group2)
        st.write(f"t-Statistic: {t_stat:.4f}")
        st.write(f"p-Value: {p_value:.4e}")
        if p_value < 0.05:
            st.success("The difference between the groups is statistically significant (p < 0.05).")
        else:
            st.info("No statistically significant difference between the groups (p ≥ 0.05).")

    # Correlation Analysis
    elif analysis_type == "Correlation":
        st.subheader("Correlation Analysis (Pearson)")
        if st.button("Explanation of Correlation Analysis"):
            st.info("""
            **Pearson Correlation**

            Pearson correlation measures the strength and direction of a linear relationship between two numerical variables.

            **Examples of Use:**
            - Relationship between height and weight.
            - Relationship between study time and test scores.

            **Interpretation of Correlation Coefficient (r):**
            - **r = 1**: Perfect positive correlation (as one variable increases, the other increases proportionally).
            - **r = -1**: Perfect negative correlation (as one variable increases, the other decreases proportionally).
            - **r = 0**: No linear relationship between the variables.

            **Prerequisites:**
            1. **Linearity**: There should be a linear relationship between the variables.
            2. **Normal Distribution**: Both variables should be approximately normally distributed.
            """)

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_columns) < 2:
            st.warning("At least two numerical columns are required for correlation analysis.")
            return
        col_x = st.selectbox("Select the first variable (X)", numeric_columns)
        col_y = st.selectbox("Select the second variable (Y)", [col for col in numeric_columns if col != col_x])
        corr, p_value = stats.pearsonr(df[col_x], df[col_y])
        st.write(f"Pearson Correlation Coefficient: {corr:.4f}")
        st.write(f"p-Value: {p_value:.4e}")
        if p_value < 0.05:
            st.success("The correlation is statistically significant (p < 0.05).")
        else:
            st.info("The correlation is not statistically significant (p ≥ 0.05).")

    # Chi²-Test
    elif analysis_type == "Chi²-Test":
        st.subheader("Chi² Test for Independence")
        if st.button("Explanation of Chi² Test"):
            st.info("""
            **Chi² Test for Independence**

            The Chi² test checks whether a relationship exists between two categorical variables.

            **Examples of Use:**
            - Is there a relationship between gender and career choice?
            - Is there a dependency between smoking (yes/no) and the occurrence of diseases (yes/no)?

            **Prerequisites:**
            1. The data must be summarized in a contingency table.
            2. Expected frequencies in each cell of the contingency table should be at least 5.

            **Interpretation:**
            - A low p-value (p < 0.05) indicates a relationship between the variables.
            - A high p-value (p ≥ 0.05) suggests that the variables are independent.
            """)

        categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
        if len(categorical_columns) < 2:
            st.warning("At least two categorical columns are required for a Chi² test.")
            return
        col_x = st.selectbox("Select the first categorical variable", categorical_columns)
        col_y = st.selectbox("Select the second categorical variable", [col for col in categorical_columns if col != col_x])
        contingency_table = pd.crosstab(df[col_x], df[col_y])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        st.write(f"Chi² Statistic: {chi2:.4f}")
        st.write(f"p-Value: {p_value:.4e}")
        if p_value < 0.05:
            st.success("The variables are statistically significantly dependent (p < 0.05).")
        else:
            st.info("There is no statistically significant dependency between the variables (p ≥ 0.05).")

    # Regression
    elif analysis_type == "Regression":
        st.subheader("Linear Regression (OLS)")
        if st.button("Explanation of Regression"):
            st.info("""
            **Linear Regression**

            Linear regression models the relationship between a target variable (Y) and one or more independent variables (X) to make predictions or understand the relationship.

            **Examples of Use:**
            - Predicting income based on years of education.
            - Relationship between advertising budget and sales.

            **Prerequisites:**
            1. **Linearity**: The relationship between the independent variables and the target variable should be linear.
            2. **Homoscedasticity**: The spread of residuals (errors) should be constant across the range of values.
            3. **Normal Distribution of Residuals**: The errors should be approximately normally distributed.
            4. **Independence of Observations**: There should be no autocorrelation between the errors.
            """)

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_columns:
            st.warning("Your dataset requires at least one numerical column.")
            return
        target_col = st.selectbox("Select the target variable (Y)", numeric_columns)
        features_possible = [col for col in numeric_columns if col != target_col]
        selected_features = st.multiselect("Select one or more features (X)", features_possible)
        if st.button("Train Regression Model"):
            if not selected_features:
                st.warning("Please select at least one feature column.")
                return
            X = df[selected_features]
            y = df[target_col]
            data = pd.concat([X, y], axis=1).dropna()
            X = data[selected_features]
            y = data[target_col]
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()
            st.write("**Regression Summary**")
            st.text(model.summary())
            st.session_state["regression_model"] = model
            st.session_state["regression_features"] = selected_features
            st.session_state["regression_target"] = target_col
            st.session_state["regression_X_const"] = X_const
            
            # Plot regression line
            if len(selected_features) == 1:
                feature = selected_features[0]
                fig, ax = plt.subplots()
                ax.scatter(data[feature], y, alpha=0.5, label="Data Points")
                x_range = np.linspace(data[feature].min(), data[feature].max(), 100)
                x_range_df = pd.DataFrame({feature: x_range})
                x_range_df_const = sm.add_constant(x_range_df)
                y_pred_line = model.predict(x_range_df_const)
                ax.plot(x_range, y_pred_line, color="red", label="Regression Line")
                ax.set_xlabel(feature)
                ax.set_ylabel(target_col)
                ax.legend()
                st.pyplot(fig)
                st.info("""
                **Interpretation of the Regression Line:**
                The red line shows how the target variable (Y) relates to the independent variable (X).
                A steeper slope indicates a stronger relationship.
                """)

            # Check for homoscedasticity
            st.subheader("Homoscedasticity (Residual Analysis)")
            data["Predicted"] = model.predict(X_const)
            data["Residuals"] = y - data["Predicted"]
            fig_residuals, ax_residuals = plt.subplots()
            ax_residuals.scatter(data["Predicted"], data["Residuals"], alpha=0.5)
            ax_residuals.axhline(0, color="red", linestyle="--")
            ax_residuals.set_xlabel("Predicted Values")
            ax_residuals.set_ylabel("Residuals")
            ax_residuals.set_title("Residual Plot: Predictions vs. Residuals")
            st.pyplot(fig_residuals)
            st.info("""
            **Interpretation of the Residual Plot:**
            - A random scatter of residuals (without a discernible pattern) indicates homoscedasticity.
            - If a pattern (e.g., funnel shape) is visible, it may indicate heteroscedasticity.
            Heteroscedasticity means that the model is less reliable in certain areas.
            """)


#########################################
# Function: Predictions                 #
#########################################
def predictions():
    st.header("Predictions")
    st.info(
        """
        In this section, you can use the previously trained regression model to make predictions for the target variable.
        """
    )
    if "regression_model" not in st.session_state:
        st.warning("Please train a regression model first under 'Statistical Analyses'.")
        return

    model = st.session_state["regression_model"]
    features = st.session_state["regression_features"]
    target_col = st.session_state["regression_target"]
    df = st.session_state["df"]
    
    input_data = {}
    for feature in features:
        col_min = float(df[feature].min())
        col_max = float(df[feature].max())
        default_val = float(df[feature].mean())
        input_data[feature] = st.number_input(
            f"Value for {feature} (Range: {col_min} to {col_max})",
            value=default_val,
            min_value=col_min,
            max_value=col_max
        )
    
    input_df = pd.DataFrame([input_data])
    
    # Ensure the constant term is correctly added
    input_df_const = sm.add_constant(input_df, has_constant='add')
    
    # Check if the columns match
    missing_cols = set(model.model.exog_names) - set(input_df_const.columns)
    for col in missing_cols:
        input_df_const[col] = 0  # Fill missing columns with 0
    
    # Adjust the order of columns
    input_df_const = input_df_const[model.model.exog_names]
    
    prediction = model.predict(input_df_const)
    st.write(f"Predicted value for {target_col}: {prediction[0]:.4f}")

#########################################
# Function: PDF Report                  #
#########################################
def pdf_report():
    st.header("PDF Report")
    st.info(
        """
        Select the contents to include in the PDF report:
        - Descriptive Statistics
        - Correlation Matrix
        - Charts (e.g., Histograms, Boxplots)
        - Results of a selected test
        """
    )

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state["df"]
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    # Checkboxes for content
    include_description = st.checkbox("Descriptive Statistics", value=True)
    include_correlation = st.checkbox("Correlation Matrix", value=True)
    include_histograms = st.checkbox("Histograms", value=False)
    include_test_results = st.checkbox("Test Results", value=True)

    # PDF Initialization
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Statistical Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    pdf.ln(10)

    # Add contents
    if include_description:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Descriptive Statistics", ln=True)
        pdf.set_font("Courier", size=10)
        pdf.multi_cell(0, 10, df.describe().to_string())
        pdf.ln(10)

    if include_correlation and numeric_columns:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Correlation Matrix", ln=True)
        pdf.set_font("Courier", size=10)
        pdf.multi_cell(0, 10, df.corr(numeric_only=True).to_string())
        pdf.ln(10)

    if include_histograms and numeric_columns:
        for col in numeric_columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
            plt.tight_layout()
            fig.savefig(f"{col}_histogram.png")
            plt.close(fig)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"Histogram of {col}", ln=True)
            pdf.image(f"{col}_histogram.png", x=10, y=40, w=180)

    if include_test_results and "last_test" in st.session_state:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Results of the Selected Test", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, st.session_state["last_test"])

    # Save PDF
    pdf_file = "report.pdf"
    pdf.output(pdf_file)

    # Download Button
    with open(pdf_file, "rb") as f:
        st.download_button(
            "Download PDF Report",
            data=f,
            file_name="Statistical_Report.pdf",
            mime="application/pdf"
        )

#########################################
# Function: Hypothesis Manager          #
#########################################
def hypothesis_manager():
    st.header("Hypothesis Manager")

    # Check if a dataset is available
    if "df" not in st.session_state:
        st.warning("Please upload a dataset before defining hypotheses.")
        return

    # Load dataset and identify columns
    df = st.session_state["df"]
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    if not numeric_columns and not categorical_columns:
        st.error("The dataset contains no valid variables.")
        return

    # Step 1: Choose the type of hypothesis
    st.subheader("Choose Hypothesis Type")
    hypothesis_type = st.radio(
        "Select the type of hypothesis:",
        ["t-Test", "Correlation", "Chi²-Test", "Regression"]
    )

    # Step 2: Hypothesis options based on selection
    if hypothesis_type == "t-Test":
        st.subheader("t-Test (Independent Samples)")
        if numeric_columns and categorical_columns:
            selected_numeric = st.selectbox("Select a numerical variable", numeric_columns, key="ttest_numeric")
            selected_category = st.selectbox("Select a categorical variable", categorical_columns, key="ttest_category")
            groups = df[selected_category].unique()
            if len(groups) != 2:
                st.warning("Exactly two groups are required for a t-Test.")
                return

            st.markdown(f"**Automatically Generated Hypotheses:**")
            st.write(f"**H₀:** The mean of '{selected_numeric}' is equal in both groups.")
            st.write(f"**H₁:** The mean of '{selected_numeric}' differs between groups.")

            if st.button("Save Hypothesis"):
                st.session_state["current_hypothesis"] = {
                    "null_hypothesis": f"The mean of '{selected_numeric}' is equal in both groups.",
                    "alt_hypothesis": f"The mean of '{selected_numeric}' differs between groups."
                }
                st.success("Hypothesis saved successfully!")

    elif hypothesis_type == "Correlation":
        st.subheader("Correlation Analysis")
        if len(numeric_columns) >= 2:
            col_x = st.selectbox("Select the first numerical variable", numeric_columns, key="corr_x")
            col_y = st.selectbox("Select the second numerical variable", [col for col in numeric_columns if col != col_x], key="corr_y")
            st.markdown(f"**Automatically Generated Hypotheses:**")
            st.write(f"**H₀:** There is no linear correlation between '{col_x}' and '{col_y}'.")
            st.write(f"**H₁:** There is a linear correlation between '{col_x}' and '{col_y}'.")

            if st.button("Save Hypothesis"):
                st.session_state["current_hypothesis"] = {
                    "null_hypothesis": f"There is no linear correlation between '{col_x}' and '{col_y}'.",
                    "alt_hypothesis": f"There is a linear correlation between '{col_x}' and '{col_y}'."
                }
                st.success("Hypothesis saved successfully!")

    elif hypothesis_type == "Chi²-Test":
        st.subheader("Chi² Test for Independence")
        if len(categorical_columns) >= 2:
            col_x = st.selectbox("Select the first categorical variable", categorical_columns, key="chi2_x")
            col_y = st.selectbox("Select the second categorical variable", [col for col in categorical_columns if col != col_x], key="chi2_y")
            st.markdown(f"**Automatically Generated Hypotheses:**")
            st.write(f"**H₀:** There is no dependency between '{col_x}' and '{col_y}'.")
            st.write(f"**H₁:** There is a dependency between '{col_x}' and '{col_y}'.")

            if st.button("Save Hypothesis"):
                st.session_state["current_hypothesis"] = {
                    "null_hypothesis": f"There is no dependency between '{col_x}' and '{col_y}'.",
                    "alt_hypothesis": f"There is a dependency between '{col_x}' and '{col_y}'."
                }
                st.success("Hypothesis saved successfully!")

    elif hypothesis_type == "Regression":
        st.subheader("Regression")
        if len(numeric_columns) >= 2:
            target_col = st.selectbox("Select the target variable (Y)", numeric_columns, key="regression_y")
            features_possible = [col for col in numeric_columns if col != target_col]
            selected_features = st.multiselect("Select independent variables (X)", features_possible, key="regression_x")

            if selected_features:
                st.markdown(f"**Automatically Generated Hypotheses:**")
                st.write(f"**H₀:** The independent variables {', '.join(selected_features)} have no effect on '{target_col}'.")
                st.write(f"**H₁:** At least one of the independent variables {', '.join(selected_features)} affects '{target_col}'.")

                if st.button("Save Hypothesis"):
                    st.session_state["current_hypothesis"] = {
                        "null_hypothesis": f"The independent variables {', '.join(selected_features)} have no effect on '{target_col}'.",
                        "alt_hypothesis": f"At least one of the independent variables {', '.join(selected_features)} affects '{target_col}'."
                    }
                    st.success("Hypothesis saved successfully!")
            else:
                st.warning("Please select at least one independent variable.")

    # Display saved hypotheses
    if "current_hypothesis" in st.session_state:
        st.markdown("---")
        st.subheader("Current Hypothesis")
        st.write(f"**H₀:** {st.session_state['current_hypothesis']['null_hypothesis']}")
        st.write(f"**H₁:** {st.session_state['current_hypothesis']['alt_hypothesis']}")

#########################################
# Main Block                            #
#########################################
if __name__ == "__main__":
    main()
