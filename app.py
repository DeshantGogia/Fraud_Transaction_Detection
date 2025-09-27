import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .fraud {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .legitimate {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .feature-importance {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('fraud_detection_xgboost_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'fraud_detection_xgboost_model.pkl' is in the same directory.")
        return None

# Feature names (must match your training data)
FEATURE_NAMES = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
    'newbalanceDest', 'isFlaggedFraud', 'balance_change_orig', 'balance_change_dest',
    'transaction_hour', 'is_merchant_dest', 'zero_balance_after', 'account_empty',
    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

def create_transaction_features(input_dict):
    """Create feature array from user input"""
    features = []
    
    # Basic features
    features.extend([
        input_dict['step'],
        input_dict['amount'],
        input_dict['oldbalanceOrg'],
        input_dict['newbalanceOrig'],
        input_dict['oldbalanceDest'],
        input_dict['newbalanceDest'],
        input_dict['isFlaggedFraud']
    ])
    
    # Calculated features
    balance_change_orig = input_dict['newbalanceOrig'] - input_dict['oldbalanceOrg']
    balance_change_dest = input_dict['newbalanceDest'] - input_dict['oldbalanceDest']
    transaction_hour = input_dict['step'] % 24
    is_merchant_dest = 1 if input_dict['nameDest'].startswith('M') else 0
    zero_balance_after = 1 if input_dict['newbalanceOrig'] == 0 else 0
    account_empty = 1 if (input_dict['oldbalanceOrg'] > 0 and input_dict['newbalanceOrig'] == 0) else 0
    
    features.extend([
        balance_change_orig,
        balance_change_dest,
        transaction_hour,
        is_merchant_dest,
        zero_balance_after,
        account_empty
    ])
    
    # Transaction type encoding
    transaction_type = input_dict['type']
    type_features = {
        'CASH_IN': [1, 0, 0, 0, 0],
        'CASH_OUT': [0, 1, 0, 0, 0],
        'DEBIT': [0, 0, 1, 0, 0],
        'PAYMENT': [0, 0, 0, 1, 0],
        'TRANSFER': [0, 0, 0, 0, 1]
    }
    features.extend(type_features[transaction_type])
    
    return np.array(features).reshape(1, -1)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Sidebar for information
    with st.sidebar:
        st.header("About")
        st.info("""
        This system detects fraudulent transactions using machine learning.
        The model was trained on historical transaction data and achieves:
        - 99.91% Accuracy
        - 99.91% ROC-AUC
        - 73% Fraud F1-Score
        """)
        
        st.header("Model Info")
        st.write("**Algorithm:** XGBoost")
        st.write("**Features:** 18 engineered features")
        st.write("**Training Data:** 8.9M transactions")
        
        st.header("Key Fraud Indicators")
        st.write("• Large transaction amounts")
        st.write("• Account emptying patterns")
        st.write("• TRANSFER transactions")
        st.write("• Zero balance after transaction")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Transaction Details")
        
        # Create form for transaction input
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                step = st.number_input("Step (Time unit)", min_value=1, max_value=744, value=100)
                amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
                type_options = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
                transaction_type = st.selectbox("Transaction Type", type_options)
                
                oldbalanceOrg = st.number_input("Origin Old Balance", min_value=0.0, value=5000.0, step=100.0)
                newbalanceOrig = st.number_input("Origin New Balance", min_value=0.0, value=4000.0, step=100.0)
            
            with col2:
                nameDest = st.text_input("Destination Name", value="C123456789")
                oldbalanceDest = st.number_input("Destination Old Balance", min_value=0.0, value=1000.0, step=100.0)
                newbalanceDest = st.number_input("Destination New Balance", min_value=0.0, value=2000.0, step=100.0)
                isFlaggedFraud = st.selectbox("Flagged as Fraud by System", [0, 1])
                transaction_hour = st.number_input("Transaction Hour", min_value=0, max_value=23, value=14)
            
            submitted = st.form_submit_button("Check for Fraud")
    
    with col2:
        st.header("Quick Analysis")
        
        if submitted:
            # Create input dictionary
            input_data = {
                'step': step,
                'amount': amount,
                'type': transaction_type,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'nameDest': nameDest,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest,
                'isFlaggedFraud': isFlaggedFraud
            }
            
            # Create features
            try:
                features = create_transaction_features(input_data)
                
                # Make prediction
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0][1]
                
                # Display results
                st.subheader("Prediction Result")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box fraud">
                        <h3>🚨 FRAUD DETECTED</h3>
                        <p><strong>Probability:</strong> {probability:.4f}</p>
                        <p><strong>Risk Level:</strong> HIGH</p>
                        <p>This transaction shows suspicious patterns.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box legitimate">
                        <h3>✅ LEGITIMATE TRANSACTION</h3>
                        <p><strong>Probability:</strong> {probability:.4f}</p>
                        <p><strong>Risk Level:</strong> LOW</p>
                        <p>This transaction appears normal.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk analysis
                st.subheader("Risk Analysis")
                
                # Create a gauge chart for probability
                st.write(f"Fraud Probability: {probability:.2%}")
                st.progress(float(probability))
                
                if probability > 0.7:
                    st.warning("High risk transaction - recommend manual review")
                elif probability > 0.3:
                    st.info("Medium risk transaction - monitor closely")
                else:
                    st.success("Low risk transaction - likely legitimate")
                
                # Feature explanation
                st.subheader("Key Factors")
                
                # Explain important features
                risk_factors = []
                if amount > 10000:
                    risk_factors.append("Large transaction amount")
                if (oldbalanceOrg > 0 and newbalanceOrig == 0):
                    risk_factors.append("Account emptied after transaction")
                if transaction_type == 'TRANSFER':
                    risk_factors.append("TRANSFER type transaction")
                if nameDest.startswith('C') and amount > 5000:
                    risk_factors.append("Large transfer to customer account")
                
                if risk_factors:
                    st.write("⚠️ **Risk factors identified:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.write("✅ No significant risk factors identified")
                    
            except Exception as e:
                st.error(f"Error processing transaction: {str(e)}")
        
        else:
            st.info("Please fill in the transaction details and click 'Check for Fraud'")
            
            # Sample transactions for testing
            st.subheader("Sample Fraud Patterns")
            st.write("Try these patterns to test the system:")
            
            sample1 = "Step: 50, Amount: 15000, Type: TRANSFER, Origin: 15000→0"
            sample2 = "Step: 200, Amount: 500, Type: PAYMENT, Origin: 1000→500"
            
            st.write(f"• {sample1}")
            st.write(f"• {sample2}")
    
    # Additional information section
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("How It Works")
        st.write("""
        1. **Input Transaction Data**: User provides transaction details
        2. **Feature Engineering**: System calculates derived features
        3. **Model Prediction**: XGBoost model analyzes patterns
        4. **Risk Assessment**: Probability score determines fraud likelihood
        5. **Decision Support**: Results help investigators prioritize reviews
        """)
    
    with col2:
        st.header("Model Performance")
        
        metrics = {
            "Accuracy": "99.91%",
            "ROC-AUC": "99.91%",
            "Fraud Precision": "58%",
            "Fraud Recall": "98%",
            "Fraud F1-Score": "73%"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)

if __name__ == "__main__":
    main()