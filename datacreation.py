import pandas as pd

def create_and_save_dataframe(data, filename="output.csv"):
    """
    Creates a pandas DataFrame from the given data and saves it to a CSV file.

    Args:
        data (dict): A dictionary containing the data for the DataFrame.
        filename (str): The name of the output CSV file.
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Successfully created and saved {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Get user input for loan application details
    print("Please enter the following details for the loan application:")
    gender = input("Gender (Male/Female): ")
    married = input("Married (Yes/No): ")
    dependents = input("Number of Dependents: ")
    education = input("Education (Graduate/Not Graduate): ")
    self_employed = input("Self Employed (Yes/No): ")
    applicant_income = int(input("Applicant Income: "))
    coapplicant_income = int(input("Coapplicant Income: "))
    loan_amount = int(input("Loan Amount: "))
    loan_amount_term = int(input("Loan Amount Term (in months): "))
    credit_history = int(input("Credit History (1 for Yes, 0 for No): "))
    property_area = input("Property Area (Urban/Rural/Semiurban): ")

    # Create a dictionary with the user's input
    loan_application_data = {
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    }

    # Create and save the DataFrame
    create_and_save_dataframe(loan_application_data)
