import datetime
from urllib.parse import unquote
import pandas as pd
import requests
from flask import Flask, json, jsonify, request
from flask_cors import CORS
from volume_based_features.VolumeBasedFeatures import VolumeBasedFeatures
from pdfs_documents.PDFReport import PDFReport
from algorithmic_trading.DailyEquity import DailyEquity

app = Flask(__name__)

@app.route("/trading/plot_daily_price_equity")
def plot_daily_price_equity():
    try:
        ticker = unquote(request.args.get("ticker"))
        equity = DailyEquity(ticker=ticker)
        data = equity.plot_price_equity()
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"Message": f"Error: {e}"}), 500
    
@app.route("/trading/plot_daily_price_equity_technical_indicators")
def plot_daily_price_equity_technical_indicators():
    try:
        ticker = unquote(request.args.get("ticker"))
        equity = DailyEquity(ticker=ticker)
        data = equity.plot_v1_technical_indicators()
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"Message": f"Error: {e}"}), 500
    
@app.route('/trading/plot_stock_volume_vs_moving_average', methods=['GET'])
def plot_stock_volume_vs_moving_average():
    try:
        tickers = request.args.get('tickers')
        start_date = request.args.get('start_date')
        end_date = request.args.get("end_date")
        filename = request.args.get("filename")
        tickers = list(tickers.split(','))
        doc = VolumeBasedFeatures(tickers=tickers, start_date=start_date, end_date=end_date)
        df = doc.volume_features_dataset()
        encoded_image, info, promptResult = doc.plot_stock_volume_vs_moving_average()
        info_string = "\n\n".join(value for item in info for value in item.values())
        info_string = info_string.strip()
        data = {
            "Visualization": encoded_image, 
            "Ticker": f'{str(tickers)}', 
            "StartDate": start_date, 
            "EndDate": end_date, 
            "Pragraph": str(info_string),
            "Interpretation": promptResult,
            "CreatedAT": str(datetime.datetime.now())
        }
        report = PDFReport(f"{filename}")
        response = report.build_pdf(data)
        return jsonify({"Message": response}), 200
    except Exception as e:
        return jsonify({'message': f'error: {(e)}'}), 500
    
@app.route('/trading/plot_daily_volume_change', methods=['GET'])
def plot_daily_volume_change():
    try:
        tickers = request.args.get('tickers')
        start_date = request.args.get('start_date')
        end_date = request.args.get("end_date")
        filename = request.args.get("filename")
        tickers = list(tickers.split(','))
        doc = VolumeBasedFeatures(tickers=tickers, start_date=start_date, end_date=end_date)
        df = doc.volume_features_dataset()
        encoded_image, info, promptResult = doc.plot_daily_volume_change()
        info_string = "\n\n".join(value for item in info for value in item.values())
        info_string = info_string.strip()
        data = {
            "Visualization": encoded_image, 
            "Ticker": f'{str(tickers)}', 
            "StartDate": start_date, 
            "EndDate": end_date, 
            "Pragraph": str(info_string),
            "Interpretation": promptResult,
            "CreatedAT": str(datetime.datetime.now())
        }
        report = PDFReport(f"{filename}")
        response = report.build_pdf(data)
        return jsonify({"Message": response}), 200
    except Exception as e:
        return jsonify({'message': f'error: {(e)}'}), 500
    
@app.route('/trading/plot_volume_surges_vs_price', methods=['GET'])
def plot_volume_surges_vs_price():
    try:
        tickers = request.args.get('tickers')
        start_date = request.args.get('start_date')
        end_date = request.args.get("end_date")
        filename = request.args.get("filename")
        tickers = list(tickers.split(','))
        doc = VolumeBasedFeatures(tickers=tickers, start_date=start_date, end_date=end_date)
        df = doc.volume_features_dataset()
        encoded_image, info, promptResult = doc.plot_volume_surges_vs_price()
        info_string = "\n\n".join(value for item in info for value in item.values())
        info_string = info_string.strip()
        data = {
            "Visualization": encoded_image, 
            "Ticker": f'{str(tickers)}', 
            "StartDate": start_date, 
            "EndDate": end_date, 
            "Pragraph": str(info_string),
            "Interpretation": promptResult,
            "CreatedAT": str(datetime.datetime.now())
        }
        report = PDFReport(f"{filename}")
        response = report.build_pdf(data)
        return jsonify({"Message": response}), 200
    except Exception as e:
        return jsonify({'message': f'error: {(e)}'}), 500
    
@app.route('/trading/plot_rsi', methods=['GET'])
def plot_rsi():
    try:
        tickers = request.args.get('tickers')
        start_date = request.args.get('start_date')
        end_date = request.args.get("end_date")
        filename = request.args.get("filename")
        tickers = list(tickers.split(','))
        doc = VolumeBasedFeatures(tickers=tickers, start_date=start_date, end_date=end_date)
        df = doc.volume_features_dataset()
        encoded_image, paragraph, promptResult = doc.plot_rsi()
        data = {
            "Visualization": encoded_image, 
            "Ticker": f'{str(tickers)}', 
            "StartDate": start_date, 
            "EndDate": end_date, 
            "Pragraph": str(paragraph),
            "Interpretation": promptResult,
            "CreatedAT": str(datetime.datetime.now())
        }
        report = PDFReport(f"{filename}")
        response = report.build_pdf(data)
        return jsonify({"Message": response}), 200
    except Exception as e:
        return jsonify({'message': f'error: {(e)}'}), 500
    
@app.route('/trading/plot_macd', methods=['GET'])
def plot_macd():
    try:
        tickers = request.args.get('tickers')
        start_date = request.args.get('start_date')
        end_date = request.args.get("end_date")
        filename = request.args.get("filename")
        tickers = list(tickers.split(','))
        doc = VolumeBasedFeatures(tickers=tickers, start_date=start_date, end_date=end_date)
        df = doc.volume_features_dataset()
        encoded_image, paragraph, promptResult = doc.plot_macd()
        data = {
            "Visualization": encoded_image, 
            "Ticker": f'{str(tickers)}', 
            "StartDate": start_date, 
            "EndDate": end_date, 
            "Pragraph": str(paragraph),
            "Interpretation": promptResult,
            "CreatedAT": str(datetime.datetime.now())
        }
        report = PDFReport(f"{filename}")
        response = report.build_pdf(data)
        return jsonify({"Message": response}), 200
    except Exception as e:
        return jsonify({'message': f'error: {(e)}'}), 500
    
@app.route('/trading/plot_volume_prediction', methods=['GET'])
def plot_volume_prediction():
    try:
        tickers = request.args.get('tickers')
        start_date = request.args.get('start_date')
        end_date = request.args.get("end_date")
        filename = request.args.get("filename")
        tickers = list(tickers.split(','))
        doc = VolumeBasedFeatures(tickers=tickers, start_date=start_date, end_date=end_date)
        df = doc.volume_features_dataset()
        encoded_image, paragraph, promptResult = doc.plot_volume_prediction()
        data = {
            "Visualization": encoded_image, 
            "Ticker": f'{str(tickers)}', 
            "StartDate": start_date, 
            "EndDate": end_date, 
            "Pragraph": str(paragraph),
            "Interpretation": promptResult,
            "CreatedAT": str(datetime.datetime.now())
        }
        report = PDFReport(f"{filename}")
        response = report.build_pdf(data)
        return jsonify({"Message": response}), 200
    except Exception as e:
        return jsonify({'message': f'error: {(e)}'}), 500
    
@app.route('/trading/plot_prophet_volume_prediction_for_multiple_stocks', methods=['GET'])
def plot_prophet_volume_prediction_for_multiple_stocks():
    try:
        tickers = request.args.get('tickers')
        start_date = request.args.get('start_date')
        end_date = request.args.get("end_date")
        filename = request.args.get("filename")
        tickers = list(tickers.split(','))
        doc = VolumeBasedFeatures(tickers=tickers, start_date=start_date, end_date=end_date)
        df = doc.volume_features_dataset()
        encoded_image, summary, promptResult = doc.plot_prophet_volume_prediction_for_multiple_stocks()
        data = {
            "Visualization": encoded_image, 
            "Ticker": f'{str(tickers)}', 
            "StartDate": start_date, 
            "EndDate": end_date, 
            "Pragraph": str(summary),
            "Interpretation": promptResult,
            "CreatedAT": str(datetime.datetime.now())
        }
        report = PDFReport(f"{filename}")
        response = report.build_pdf(data)
        return jsonify({"Message": response}), 200
    except Exception as e:
        return jsonify({'message': f'error: {(e)}'}), 500
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)