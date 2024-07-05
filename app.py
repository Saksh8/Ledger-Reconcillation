from flask import Flask, request, render_template, send_file
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def upload_files():
    return render_template('upload.html')

@app.route('/reconcile', methods=['POST'])
def reconcile_files():
    ledger_file = request.files['ledger']
    vendor_file = request.files['vendor']
    
    ledger_path = os.path.join(app.config['UPLOAD_FOLDER'], ledger_file.filename)
    vendor_path = os.path.join(app.config['UPLOAD_FOLDER'], vendor_file.filename)
    
    ledger_file.save(ledger_path)
    vendor_file.save(vendor_path)
    
    # Import the reconciliation logic from the converted script
    from Reconcillation_Logic import reconcile_ledgers

    # Read the CSV files
    ledger_df = pd.read_csv(ledger_path)
    vendor_df = pd.read_csv(vendor_path)
    
    # Perform reconciliation logic
    reconciled_df = reconcile_ledgers(ledger_df, vendor_df)
    
    # Save the result to a new CSV file
    output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'reconciled.csv')
    reconciled_df.to_csv(output_file, index=False)
    
    return send_file(output_file, as_attachment=True, download_name='reconciled.csv')

if __name__ == '__main__':
    app.run(debug=True)
