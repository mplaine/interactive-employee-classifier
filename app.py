"""
Interactive Employee Classifier


Built with Streamlit in Python.

Note: Familiarize yourself with Streamlit data flow,
https://docs.streamlit.io/get-started/fundamentals/main-concepts#data-flow
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

import altair as alt
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------

__author__ = "Markku Laine"
__email__ = "markku.laine@gmail.com"
__version__ = "1.0"


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------

# Constants
DATASETS_DIR = Path("datasets")
FORM_BUTTON_LABEL_ENABLED = "Train model"
FORM_BUTTON_LABEL_DISABLED = "Training model..."


def main():
    initialize_session()
    render_ui()


def initialize_session():
    if "form_submit_button_label" not in st.session_state:
        st.session_state["form_submit_button_label"] = FORM_BUTTON_LABEL_ENABLED
    if "form_submit_button_disabled" not in st.session_state:
        st.session_state["form_submit_button_disabled"] = False
    if "y_test" not in st.session_state:
        st.session_state["y_test"] = None
    if "y_pred" not in st.session_state:
        st.session_state["y_pred"] = None
    if "feature_names" not in st.session_state:
        st.session_state["feature_names"] = None
    if "feature_importances" not in st.session_state:
        st.session_state["feature_importances"] = None
    if "show_evaluation_results" not in st.session_state:
        st.session_state["show_evaluation_results"] = False
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {
            "previous_scores" : {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None
            },
            "present_scores" : {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None
            }
        }


def handle_form_submit_button_click():
    st.session_state["form_submit_button_disabled"] = True
    st.session_state["form_submit_button_label"] = FORM_BUTTON_LABEL_DISABLED


def render_ui():
    # Main
    st.title("Interactive Employee Classifier")
    st.write(f"`Version {__version__}`")
    st.write("Want to predict employee attrition using the features from a HR dataset? Well, look no further! This interactive tool allows you to build and evaluate different machine learning classification models exactly for that purpose.")
    main_container = st.container()

    # Sidebar
    with st.sidebar:
        # Header
        st.image(image="static/streamlit_logo.svg", width=200)
        st.title(f"Interactive Employee Classifier `version {__version__}`")

        # Dataset
        st.header("Dataset")
        dataset = st.selectbox(label="Choose CSV file", options=("hr_dataset_v1.csv", "hr_dataset_v2.csv"), index=0, key="dataset")
        df = load_data(DATASETS_DIR / str(dataset))
        X_train, X_test, y_train, y_test = split_data(df)
        if st.checkbox(label="Preview dataset", value=False, key="preview_dataset"):
            with main_container:
                st.header("Dataset Preview")
                st.dataframe(data=df, use_container_width=True)

        # Model
        st.header("Model")
        algorithm = st.selectbox(label="Choose algorithm", options=("Decision Tree", "Random Forest", "XGBoost"), index=0, key="algorithm")

        # Form
        form = st.form(key="data_modeling_form", border=False)
        with form:
            # Hyperparameters
            st.header("Hyperparameters")
            with st.expander("Fine-tune model", expanded=False):
                if algorithm == "Decision Tree":
                    criterion = st.selectbox(label="criterion", options=("entropy", "gini", "log_loss"), index=1, help="The function to measure the quality of a split.")
                    max_depth = st.number_input(label="max_depth", min_value=1, max_value=None, value=None, step=1, placeholder="None", help="The maximum depth of the tree.")
                    max_features = st.selectbox(label="max_features", options=(None, "log2", "sqrt"), index=0, help="The number of features to consider when looking for the best split.")
                    min_samples_leaf = st.number_input(label="min_samples_leaf", min_value=1, max_value=None, value=1, step=1, help="The minimum number of samples required to be at a leaf node.")
                    min_samples_split = st.number_input(label="min_samples_split", min_value=2, max_value=None, value=2, step=1, help="The minimum number of samples required to split an internal node.")
                elif algorithm == "Random Forest":
                    bootstrap = st.toggle(label="bootstrap", value=True, help="Whether bootstrap samples are used when building trees.")
                    criterion = st.selectbox(label="criterion", options=("gini", "entropy", "log_loss"), index=0, help="The function to measure the quality of a split.")
                    max_depth = st.number_input(label="max_depth", min_value=1, max_value=None, value=None, step=1, placeholder="None", help="The maximum depth of the tree.")
                    max_features = st.selectbox(label="max_features", options=(None, "log2", "sqrt"), index=2, help="The number of features to consider when looking for the best split.")
                    max_samples = st.number_input(label="max_samples", min_value=1, max_value=X_train.shape[0], value=None, step=1, placeholder="None", help="If bootstrap is True, the number of samples to draw from X to train each base estimator.")
                    min_samples_leaf = st.number_input(label="min_samples_leaf", min_value=1, max_value=None, value=1, step=1, help="The minimum number of samples required to be at a leaf node.")
                    min_samples_split = st.number_input(label="min_samples_split", min_value=2, max_value=None, value=2, step=1, help="The minimum number of samples required to split an internal node.")
                    n_estimators = st.number_input(label="n_estimators", min_value=1, max_value=500, value=100, step=1, help="The number of trees in the forest.")
                elif algorithm == "XGBoost":
                    max_depth = st.number_input(label="max_depth", min_value=0, max_value=None, value=6, step=1, help="Maximum tree depth for base learners.")
                    min_child_weight = st.number_input(label="min_child_weight", min_value=0, max_value=None, value=1, step=1, help="Minimum sum of instance weight(hessian) needed in a child.")
                    subsample = st.slider(label="subsample", min_value=0.01, max_value=1.0, value=1.0, step=0.01, help="Subsample ratio of the training instance.")
                    colsample_bytree = st.slider(label="colsample_bytree", min_value=0.01, max_value=1.0, value=1.0, step=0.01, help="Subsample ratio of columns when constructing each tree.")
                    learning_rate = st.slider(label="learning_rate", min_value=0.01, max_value=1.0, value=0.3, step=0.01, help="Boosting learning rate (xgb’s \"eta\")")
                    n_estimators = st.number_input(label="n_estimators", min_value=1, max_value=500, value=100, step=1, help="Number of boosting rounds.")

            # Form submit button
            st.form_submit_button(label=st.session_state["form_submit_button_label"], type="primary", use_container_width=True, disabled=st.session_state["form_submit_button_disabled"], on_click=handle_form_submit_button_click)

        # Footer
        st.markdown("""
            ---
            Created with ❤️ by [Markku Laine](https://markkulaine.com).            
        """)

        # Form submit button click handler when disabled
        if st.session_state["form_submit_button_disabled"]:
            st.session_state["show_evaluation_results"] = True

            # Create model instance
            if algorithm == "Decision Tree":
                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=42) # type: ignore
            elif algorithm == "Random Forest":
                model = RandomForestClassifier(bootstrap=bootstrap, criterion=criterion, max_depth=max_depth, max_features=max_features, max_samples=max_samples, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators, n_jobs=-1, random_state=42) # type: ignore
            elif algorithm == "XGBoost":
                model = XGBClassifier(colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight, n_estimators=n_estimators, subsample=subsample, objective="binary:logistic", n_jobs=-1, random_state=42) # type: ignore

            # Fit model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Update session state
            st.session_state["y_test"] = y_test
            st.session_state["y_pred"] = y_pred
            st.session_state["feature_names"] = model.feature_names_in_
            st.session_state["feature_importances"] = model.feature_importances_
            st.session_state["metrics"]["previous_scores"]["accuracy"] = st.session_state["metrics"]["present_scores"]["accuracy"]
            st.session_state["metrics"]["previous_scores"]["precision"] = st.session_state["metrics"]["present_scores"]["precision"]
            st.session_state["metrics"]["previous_scores"]["recall"] = st.session_state["metrics"]["present_scores"]["recall"]
            st.session_state["metrics"]["previous_scores"]["f1"] = st.session_state["metrics"]["present_scores"]["f1"]
            st.session_state["metrics"]["present_scores"]["accuracy"] = accuracy_score(y_test, y_pred)
            st.session_state["metrics"]["present_scores"]["precision"] = precision_score(y_test, y_pred)
            st.session_state["metrics"]["present_scores"]["recall"] = recall_score(y_test, y_pred)
            st.session_state["metrics"]["present_scores"]["f1"] = f1_score(y_test, y_pred)
            st.session_state["form_submit_button_disabled"] = False
            st.session_state["form_submit_button_label"] = FORM_BUTTON_LABEL_ENABLED
            st.rerun() # to update submit button label and status

    # Show evaluation results
    if st.session_state["show_evaluation_results"]:
        show_evaluation_results(main_container)


@st.cache_data
def load_data(filepath: Path) -> pd.DataFrame:
    # Load the dataset
    data = pd.read_csv(filepath)

    # Return the dataframe
    return data


@st.cache_data
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Select predictor (X) and target (y) variables
    X = df.drop(["left"], axis=1)
    y = df["left"]

    # Split into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Return the sets
    return X_train, X_test, y_train, y_test


def show_evaluation_results(main_container):
    with main_container:
        st.header("Evaluation Results")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance Metrics", "Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importances"])

        with tab1:
            plot_performance_metrics()

        with tab2:
            plot_confusion_matrix()

        with tab3:
            plot_roc_curve()

        with tab4:
            plot_precision_recall_curve()

        with tab5:
            plot_feature_importances()


def plot_performance_metrics():
    st.subheader("Performance Metrics")

    # Retrieve session state
    previous_accuracy = st.session_state["metrics"]["previous_scores"]["accuracy"]
    previous_precision = st.session_state["metrics"]["previous_scores"]["precision"]
    previous_recall = st.session_state["metrics"]["previous_scores"]["recall"]
    previous_f1 = st.session_state["metrics"]["previous_scores"]["f1"]
    present_accuracy = st.session_state["metrics"]["present_scores"]["accuracy"]
    present_precision = st.session_state["metrics"]["present_scores"]["precision"]
    present_recall = st.session_state["metrics"]["present_scores"]["recall"]
    present_f1 = st.session_state["metrics"]["present_scores"]["f1"]

    # Compute deltas
    accuracy_delta = None if previous_accuracy is None else "{:.4f}".format(present_accuracy - previous_accuracy)
    precision_delta = None if previous_precision is None else "{:.4f}".format(present_precision - previous_precision)
    recall_delta = None if previous_recall is None else "{:.4f}".format(present_recall - previous_recall)
    f1_delta = None if previous_f1 is None else "{:.4f}".format(present_f1 - previous_f1)

    # Render performance metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Accuracy score", value=f"{present_accuracy:.4f}", delta=accuracy_delta, help="The proportion of data points predicted correctly out of all the data points.")
    col2.metric(label="Precision score", value=f"{present_precision:.4f}", delta=precision_delta, help="The proportion of data points predicted as True that are actually True.")
    col3.metric(label="Recall score", value=f"{present_recall:.4f}", delta=recall_delta, help="The proportion of data points predicted as True out of all the data points that are actually True.")
    col4.metric(label="F1 score", value=f"{present_f1:.4f}", delta=f1_delta, help="The harmonic mean of precision and recall.")


def plot_confusion_matrix():
    st.subheader("Confusion Matrix")

    # Retrieve session state
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]

    # Update labels
    labels = ["Stayed", "Left"]
    label_mapping = {0: labels[0], 1: labels[1]}
    y_test_transformed = [label_mapping[label] for label in y_test]
    y_pred_transformed = [label_mapping[label] for label in y_pred]

    # Compute confusion matrix
    cm = confusion_matrix(y_test_transformed, y_pred_transformed, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Render confusion matrix
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d", cmap="Reds")
    st.pyplot(fig)


def plot_roc_curve():
    st.subheader("ROC Curve")

    # Retrieve session state
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]
    algorithm = st.session_state["algorithm"]

    # Update labels
    labels = ["Stayed", "Left"]
    label_mapping = {0: labels[0], 1: labels[1]}
    y_test_transformed = [label_mapping[label] for label in y_test]

    # Compute ROC curve
    disp = RocCurveDisplay.from_predictions(y_true=y_test_transformed, y_pred=y_pred, name=algorithm, pos_label=labels[1])

    # Render ROC curve
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)


def plot_precision_recall_curve():
    st.subheader("Precision-Recall Curve")

    # Retrieve session state
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]
    algorithm = st.session_state["algorithm"]

    # Update labels
    labels = ["Stayed", "Left"]
    label_mapping = {0: labels[0], 1: labels[1]}
    y_test_transformed = [label_mapping[label] for label in y_test]

    # Compute Precision-Recall curve
    disp = PrecisionRecallDisplay.from_predictions(y_true=y_test_transformed, y_pred=y_pred, name=algorithm, pos_label=labels[1])

    # Render Precision-Recall curve
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)


def plot_feature_importances():
    st.subheader("Feature Importances")

    # Retrieve session state
    feature_names = st.session_state["feature_names"]
    feature_importances = st.session_state["feature_importances"]

    # Create feature importances dataframe
    df_fi = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Render feature importances
    chart = alt.Chart(df_fi).mark_bar(color=st.get_option("theme.primaryColor")).encode(x="Importance", y=alt.Y("Feature", sort="-x", axis=alt.Axis(labelLimit=300)))
    st.altair_chart(chart, theme="streamlit", use_container_width=True)


if __name__ == '__main__':
    st.set_page_config(page_title="Interactive Employee Classifier", initial_sidebar_state="expanded")
    main()
