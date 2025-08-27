import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import sys

try:
    import talib
except ImportError:
    messagebox.showerror("Erreur", "TA-Lib n'est pas installé. Installez-le avec 'pip install TA-Lib' et assurez-vous que la bibliothèque C de TA-Lib est disponible.")
    talib = None

class PredictTechApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PREDICTECH - Prédicteur de Prix d'Actions")
        self.root.geometry("1200x800")
        self.style = ttk.Style("flatly")

        # Gestionnaire de fermeture pour arrêter le processus
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.features = ['Close', 'Volume', 'SMA20', 'RSI']
        self.scalers = {feature: MinMaxScaler(feature_range=(0, 1)) for feature in self.features}
        self.model = None
        self.full_data = None
        self.available_start = None
        self.available_end = None

        self.create_widgets()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Désactiver l'interactivité
        self.canvas.toolbar_visible = False  # Masquer la barre d'outils
        self.canvas.mpl_connect('button_press_event', lambda event: None)  # Désactiver les clics
        self.canvas.mpl_connect('motion_notify_event', lambda event: None)  # Désactiver les mouvements de souris

    def on_closing(self):
        """Termine le processus Python lorsque la fenêtre est fermée"""
        self.root.destroy()
        sys.exit()

    def create_widgets(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)

        ttk.Button(control_frame, text="Charger Données", command=self.load_data, style="primary.TButton").pack(side=tk.LEFT, padx=5)

        date_frame = ttk.LabelFrame(control_frame, text="Période", padding=5)
        date_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(date_frame, text="Date Début (AAAA-MM-JJ):").grid(row=0, column=0, padx=5, pady=2)
        self.start_date_entry = ttk.Entry(date_frame, width=12)
        self.start_date_entry.grid(row=0, column=1, padx=5, pady=2)
        self.start_date_entry.insert(0, "2018-01-01")

        ttk.Label(date_frame, text="Début Prédiction:").grid(row=0, column=2, padx=5, pady=2)
        self.pred_start_entry = ttk.Entry(date_frame, width=12)
        self.pred_start_entry.grid(row=0, column=3, padx=5, pady=2)
        self.pred_start_entry.insert(0, "2020-01-01")

        ttk.Label(date_frame, text="Date Fin:").grid(row=0, column=4, padx=5, pady=2)
        self.end_date_entry = ttk.Entry(date_frame, width=12)
        self.end_date_entry.grid(row=0, column=5, padx=5, pady=2)
        self.end_date_entry.insert(0, "2023-03-31")

        param_frame = ttk.LabelFrame(control_frame, text="Paramètres du Modèle", padding=5)
        param_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(param_frame, text="Unités LSTM:").grid(row=0, column=0, padx=5, pady=2)
        self.units_entry = ttk.Entry(param_frame, width=6)
        self.units_entry.grid(row=0, column=1, padx=5, pady=2)
        self.units_entry.insert(0, "500")

        ttk.Label(param_frame, text="Jours de Retour:").grid(row=1, column=0, padx=5, pady=2)
        self.lookback_entry = ttk.Entry(param_frame, width=6)
        self.lookback_entry.grid(row=1, column=1, padx=5, pady=2)
        self.lookback_entry.insert(0, "30")

        ttk.Button(control_frame, text="Entraîner & Prédire", command=self.train_and_predict, style="success.TButton").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Réinitialiser", command=self.reset_app, style="danger.TButton").pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Prêt")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

        self.progress = ttk.Progressbar(self.root, mode='indeterminate', style="success.Striped.TProgressbar")
        self.progress.pack(fill=tk.X, padx=10, pady=5)

    def reset_app(self):
        self.full_data = None
        self.available_start = None
        self.available_end = None
        self.model = None
        
        self.start_date_entry.delete(0, tk.END)
        self.start_date_entry.insert(0, "2018-01-01")
        self.pred_start_entry.delete(0, tk.END)
        self.pred_start_entry.insert(0, "2020-01-01")
        self.end_date_entry.delete(0, tk.END)
        self.end_date_entry.insert(0, "2023-03-31")
        self.units_entry.delete(0, tk.END)
        self.units_entry.insert(0, "500")
        self.lookback_entry.delete(0, tk.END)
        self.lookback_entry.insert(0, "30")
        
        self.ax.clear()
        self.ax.set_title('Prix Historique des Actions', fontsize=14)
        self.ax.set_xlabel('Date', fontsize=12)
        self.ax.set_ylabel('Prix (€)', fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.canvas.draw()
        
        self.status_var.set("Prêt")
        self.progress.stop()

    def validate_dates(self, start_date, pred_start, end_date, lookback_days):
        try:
            start = pd.to_datetime(start_date)
            pred = pd.to_datetime(pred_start)
            end = pd.to_datetime(end_date)
            if start >= pred:
                raise ValueError("La date de début doit être antérieure à la date de début de prédiction")
            if pred >= end:
                raise ValueError("La date de début de prédiction doit être antérieure à la date de fin")
            if (end - start).days < lookback_days:
                raise ValueError(f"La période sélectionnée doit être d'au moins {lookback_days} jours")
            if self.available_start and self.available_end:
                if start < self.available_start:
                    raise ValueError(f"La date de début ne peut pas être antérieure à {self.available_start.strftime('%Y-%m-%d')}")
                if end > self.available_end:
                    raise ValueError(f"La date de fin ne peut pas être postérieure à {self.available_end.strftime('%Y-%m-%d')}")
            return True
        except ValueError as e:
            messagebox.showerror("Erreur de Date", str(e))
            return False

    def load_data(self):
        if talib is None:
            messagebox.showerror("Erreur", "TA-Lib n'est pas disponible.")
            return
        file_path = filedialog.askopenfilename(filetypes=[("Fichiers CSV", "*.csv")])
        if file_path:
            try:
                self.progress.start()
                self.status_var.set("Chargement des données...")
                self.root.update()
                self.full_data = pd.read_csv(file_path, parse_dates=['Date'], index_col=['Date'], usecols=['Date', 'Close', 'Volume'])
                self.full_data.sort_index(inplace=True)
                self.available_start = self.full_data.index.min()
                self.available_end = self.full_data.index.max()
                self.status_var.set(f"Données chargées de {self.available_start.strftime('%Y-%m-%d')} à {self.available_end.strftime('%Y-%m-%d')}")
                self.full_data['SMA20'] = talib.SMA(self.full_data['Close'], timeperiod=20)
                self.full_data['RSI'] = talib.RSI(self.full_data['Close'], timeperiod=14)
                self.full_data = self.full_data.asfreq('D').ffill()
                self.full_data.dropna(inplace=True)
                messagebox.showinfo("Succès", f"{len(self.full_data)} enregistrements chargés")
                self.plot_data()
            except Exception as e:
                messagebox.showerror("Erreur", f"Échec du chargement : {str(e)}")
            finally:
                self.progress.stop()
                self.root.update()

    def plot_data(self):
        self.ax.clear()
        self.ax.plot(self.full_data.index, self.full_data['Close'], label='Prix de Clôture', color='blue')
        self.ax.set_title('Prix Historique des Actions', fontsize=14)
        self.ax.set_xlabel('Date', fontsize=12)
        self.ax.set_ylabel('Prix (€)', fontsize=12)
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.ax.xaxis.set_major_locator(mdates.MonthLocator())
        self.fig.autofmt_xdate()
        self.canvas.draw()

    def prepare_data(self):
        if talib is None:
            messagebox.showerror("Erreur", "TA-Lib n'est pas disponible.")
            return None, None, None, None
        try:
            start_date = self.start_date_entry.get()
            pred_start = self.pred_start_entry.get()
            end_date = self.end_date_entry.get()
            self.lookback_days = int(self.lookback_entry.get())
            if not (10 <= self.lookback_days <= 100):
                raise ValueError("Les jours de retour doivent être entre 10 et 100")
            if not self.validate_dates(start_date, pred_start, end_date, self.lookback_days):
                return None, None, None, None
            
            extended_pred_start = pd.to_datetime(pred_start) - pd.Timedelta(days=self.lookback_days)
            if extended_pred_start < pd.to_datetime(start_date):
                extended_pred_start = pd.to_datetime(start_date)
            
            period_data = self.full_data.loc[start_date:end_date, self.features].copy()
            self.train_data = period_data.loc[start_date:pred_start]
            self.extended_test_data = period_data.loc[extended_pred_start:end_date]
            
            if len(self.train_data) < self.lookback_days:
                raise ValueError(f"Pas assez de données d'entraînement. Besoin de {self.lookback_days} jours, mais seulement {len(self.train_data)} disponibles")
            if len(self.extended_test_data) < self.lookback_days:
                raise ValueError(f"Pas assez de données de test. Besoin d'au moins {self.lookback_days} jours, mais seulement {len(self.extended_test_data)} disponibles")
            
            train_scaled = np.zeros((len(self.train_data), len(self.features)))
            extended_test_scaled = np.zeros((len(self.extended_test_data), len(self.features)))
            
            for idx, feature in enumerate(self.features):
                scaler = self.scalers[feature]
                train_scaled[:, idx] = scaler.fit_transform(self.train_data[[feature]]).flatten()
                extended_test_scaled[:, idx] = scaler.transform(self.extended_test_data[[feature]]).flatten()
            
            X_train, y_train = self.create_sequences(train_scaled, self.features.index('Close'))
            X_test, y_test = self.create_sequences(extended_test_scaled, self.features.index('Close'))
            
            return X_train, y_train, X_test, y_test
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            return None, None, None, None

    def create_sequences(self, data, target_idx):
        num_samples = len(data) - self.lookback_days
        if num_samples <= 0:
            return np.array([]), np.array([])
        X = np.array([data[i:i+self.lookback_days] for i in range(num_samples)])
        y = data[self.lookback_days:, target_idx]
        return X, y

    def build_model(self, units, num_features):
        model = Sequential([
            Input(shape=(self.lookback_days, num_features)),
            LSTM(units, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train_and_predict(self):
        if self.full_data is None:
            messagebox.showerror("Erreur", "Veuillez charger les données d'abord")
            return
        try:
            units = int(self.units_entry.get())
            if units <= 0:
                raise ValueError("Les unités LSTM doivent être un entier positif")
        except ValueError:
            messagebox.showerror("Erreur", "Valeur des unités LSTM invalide")
            return
        
        self.progress.start()
        self.status_var.set("Entraînement du modèle...")
        self.root.update()
        
        try:
            X_train, y_train, X_test, y_test = self.prepare_data()
            if X_train is None or talib is None:
                return
            
            self.model = self.build_model(units, len(self.features))
            self.model.fit(X_train, y_train, batch_size=32, epochs=50, 
                           callbacks=[EarlyStopping(monitor='loss', patience=5)], verbose=0)
            
            self.status_var.set("Prédiction en cours...")
            self.root.update()
            
            test_predictions = self.model.predict(X_test, verbose=0, batch_size=32)
            test_predictions = self.scalers['Close'].inverse_transform(test_predictions)
            actual_values = self.scalers['Close'].inverse_transform(y_test.reshape(-1, 1))
            
            pred_start = pd.to_datetime(self.pred_start_entry.get())
            extended_pred_start = pred_start - pd.Timedelta(days=self.lookback_days)
            if extended_pred_start < pd.to_datetime(self.start_date_entry.get()):
                extended_pred_start = pd.to_datetime(self.start_date_entry.get())
            
            pred_dates = self.extended_test_data.index[self.lookback_days:]
            
            if len(pred_dates) != len(test_predictions):
                raise ValueError(f"Longueur des prédictions ({len(test_predictions)}) ne correspond pas à la plage de dates ({len(pred_dates)})")
            
            results = pd.DataFrame({
                'Date': pred_dates,
                'Réel': actual_values.flatten(),
                'Prédit': test_predictions.flatten()
            })
            
            # Filtrer les résultats pour commencer à pred_start
            results = results[results['Date'] >= pred_start]
            if results.empty:
                raise ValueError("Aucune donnée prédite après le filtrage. Vérifiez les dates et le nombre de jours de retour.")
            
            self.status_var.set("Prédiction terminée.")
            self.plot_results(results)
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur s'est produite : {str(e)}")
            self.status_var.set("Erreur survenue")
        finally:
            self.progress.stop()
            self.root.update()

    def plot_results(self, results):
        self.ax.clear()
        full_period_data = self.full_data.loc[self.start_date_entry.get():self.end_date_entry.get()]
        pred_start = pd.to_datetime(self.pred_start_entry.get())
        extended_pred_start = pred_start - pd.Timedelta(days=self.lookback_days)
        if extended_pred_start < pd.to_datetime(self.start_date_entry.get()):
            extended_pred_start = pd.to_datetime(self.start_date_entry.get())
        
        # Training data (blue line)
        training_data = full_period_data[full_period_data.index < pred_start]
        self.ax.plot(training_data.index, training_data['Close'], 
                     label='Prix Réel (Entraînement)', color='blue')
        
        # Test data (green line) - Commencer à pred_start
        test_data = full_period_data[full_period_data.index >= pred_start]
        self.ax.plot(test_data.index, test_data['Close'], 
                     label='Prix Réel (Test)', color='green')
        
        # Predictions (red dashed line) - Aligner avec test_data.index
        self.ax.plot(results['Date'], results['Prédit'], 
                     label='Prix Prédit', color='red', linestyle='--')
        
        self.ax.set_title('Prédiction des Prix des Actions', fontsize=14)
        self.ax.set_xlabel('Date', fontsize=12)
        self.ax.set_ylabel('Prix (€)', fontsize=12)
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.ax.xaxis.set_major_locator(mdates.MonthLocator())
        self.fig.autofmt_xdate()
        self.canvas.draw()

if __name__ == "__main__":
    root = ttk.Window(themename="flatly")
    app = PredictTechApp(root)
    root.mainloop()