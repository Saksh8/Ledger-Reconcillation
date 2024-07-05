### Reconciliation Web Application with Flask

This Flask application reconciles ledger and vendor data files uploaded by users. It performs reconciliation based on different transaction types (invoices, payments, notes) and generates a reconciled output file.

#### Features
- **File Upload**: Users can upload their ledger and vendor CSV files.
- **Reconciliation Logic**: Automatically reconciles transactions based on dates, amounts, and transaction types.
- **Output**: Generates a CSV file containing reconciled transactions.

#### Setup and Installation
1. **Dependencies**:
   - Ensure you have Python installed.
   - Install Flask and pandas:
     ```
     pip install Flask pandas
     ```

2. **Running the Application**:
   - Clone or download the repository.
   - Navigate to the project directory in the terminal.
   - Run the Flask application:
     ```
     python app.py
     ```
   - The application will run on `http://localhost:5000`.

#### Usage
1. **Upload Files**:
   - Access the application through a web browser.
   - Click on the upload link to select your ledger and vendor CSV files.

2. **Reconciliation Process**:
   - After uploading files, click on the 'Reconcile' button.
   - The application will process the files and generate a reconciled CSV file.

3. **Download Reconciled File**:
   - Once processing is complete, the reconciled CSV file will be available for download.

#### Example
- Sample CSV files (`ledger.csv`, `vendor.csv`) are provided in the `uploads` folder for testing purposes.

#### Structure
- **app.py**: Flask application script handling file upload and reconciliation logic.
- **upload.html**: HTML template for file upload form.
- **Reconcillation_Logic.py**: Python script containing the reconciliation functions.

#### Notes
- Ensure CSV files follow the expected format with appropriate headers (`Date`, `Net Amount`, `TYPE`, etc.).
- Adjust reconciliation logic in `Reconcillation_Logic.py` to suit specific business rules or requirements.

### Reconciliation Logic Explanation

#### 1. Introduction
The project aims to match the ledger entries to vendor entries based on different feature columns, such as Sales and Purchase, Payment and Receipt, and Credit Note and Debit Note. 

#### 2. Preprocessing
Explain the preprocessing steps:
- Data type conversion (`astype(float)` and `pd.to_datetime`).
- Filtering data into different transaction types (invoices, payments, notes).

#### 3. Matching Logic
Describe the matching logic used:
- **TF-IDF Vectorization**: Vectorizing invoice numbers for cosine similarity.
- **Cosine Similarity**: Calculating similarity scores between invoice numbers present in our ledger and vendor ledger to find the best match 
- **Amount Difference**: Calculating and comparing differences in transaction amounts corresponding to two matched invoices.

#### 4. Types of Reconciliation
Detail how reconciliation is performed for different transaction types:
- **Invoices**: Matching purchase and sales invoices.
- **Payments and Receipts**: Checking for matches based on date and amount.
- **Credit and Debit Notes**: Matching credit and debit notes, considering amount differences.

#### 5. Remarks and Duplicates
Explain how remarks are added:
- Adding remarks based on match scores and amount differences. If the match score is found to be 1 and the transaction amount is less than 3Rs, then Sales/Purchase Invoices or Debit Notes/Credit Notes are considered to be matched.
- In the case of Payment and Receipt, a "Payment Match" remark is added if the net amount sum in a particular month  in our ledger matches the net amount sum in that particular month with the vendor.
- Handling duplicate invoices to avoid double-counting.

#### 6. Final Reconciliation
- Consolidating matched, mismatched, and duplicate entries.
- Keeping remaining entries that do not match any criteria.

#### 7. Usage
- The reconcillation_logic.py file is deployed using the app.py flask web application.
- The output is a reconciled csv file returning the result of reconciliation.

#### 7. Dependencies
Included in requirement.txt file.

#### 10. Future Enhancements
The project could be made more robust by handling more matching criteria such as TDS Note accounting; Time complexity could be reduced using MLOPS.NLP could be used to extract important information from the contextual data present in the Particulars section of the Ledger.



---

