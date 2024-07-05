Automation of Ledger Vendor Reconciliation

---

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
- Adding remarks based on match scores and amount differences.If the match score is found to be 1 and the transaction amount is less than 3Rs, then Sales/Purchase Invoices or Debit Notes/Credit Notes are considered to be matched.
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

