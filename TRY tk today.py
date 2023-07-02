import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split

def read_table():
    sg.set_options(auto_size_buttons=True)
    layout = [
        [sg.Text('Dataset (a CSV file)', size=(16, 1)), sg.InputText(), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),("Text Files", "*.txt")))],
        [sg.Submit(), sg.Cancel()]
    ]

    window1 = sg.Window('Input file', layout)
    try:
        event, values = window1.read()
        window1.close()
    except:
        window1.close()
        return
    
    filename = values[0]
    
    if filename == '':
        return

    data = []
    header_list = []

    if filename is not None:
        fn = filename.split('/')[-1]
        try:                     
            if colnames_checked:
                df = pd.read_csv(filename, sep=',', engine='python')
                # Uses the first row (which should be column names) as columns names
                header_list = list(df.columns)
                # Drops the first row in the table (otherwise the header names and the first row will be the same)
                data = df[1:].values.tolist()
            else:
                df = pd.read_csv(filename, sep=',', engine='python', header=None)
                # Creates columns names for each column ('column0', 'column1', etc)
                header_list = ['column' + str(x) for x in range(len(df.iloc[0]))]
                df.columns = header_list
                # read everything else into a list of rows
                data = df.values.tolist()
            # NaN drop?
            if dropnan_checked:
                df = df.dropna()
                data = df.values.tolist()
            window1.close()
            return (df,data, header_list,fn)
        except:
            sg.popup_error('Error reading file')
            window1.close()
            return

def show_table(data, header_list, fn):    
    layout = [
        [sg.Table(values=data,
                  headings=header_list,
                  font='Helvetica',
                  pad=(25,25),
                  display_row_numbers=False,
                  auto_size_columns=True,
                  num_rows=min(25, len(data)))]
    ]

    window = sg.Window(fn, layout, grab_anywhere=False)
    event, values = window.read()
    window.close()

def show_stats(df):
    stats = df.describe().T
    header_list = list(stats.columns)
    data = stats.values.tolist()
    for i,d in enumerate(data):
        d.insert(0,list(stats.index)[i])
    header_list=['Feature']+header_list
    layout = [
        [sg.Table(values=data,
                  headings=header_list,
                  font='Helvetica',
                  pad=(10,10),
                  display_row_numbers=False,
                  auto_size_columns=True,
                  num_rows=min(25, len(data)))]
    ]

    window = sg.Window("Statistics", layout, grab_anywhere=False)
    event, values = window.read()
    window.close()

def sklearn_model(output_var, algorithm, X_train, y_train, X_test, y_test):
    if algorithm == 'Linear Regression':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif algorithm == 'Random Forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=0)
    elif algorithm == 'Decision Tree':
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(max_depth=10, random_state=0)
    elif algorithm == 'KNeighbors Regressor':
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=5)
    elif algorithm == 'lasso Regression':
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1, max_iter=10000)
    elif algorithm == 'Ridge Regression':
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.1, max_iter=10000)
        
    else:
        return None, None, None, None, None

    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    accuracy_train = model.score(X_train, y_train)
    accuracy_test = model.score(X_test, y_test)

    return model, predictions_train, predictions_test, accuracy_train, accuracy_test


def plot_predictions(predictions, actual_data):
    fig, ax = plt.subplots()

    years_actual = np.arange(1993, 2022)  # Years for actual data

    actual_data = actual_data[:len(years_actual)]  # Limit actual data to match the length of years_actual
    predictions = predictions[:len(years_actual)]  # Limit predictions to match the length of years_actual

    ax.plot(years_actual, actual_data, label='Actual Data', color='blue')
    ax.plot(years_actual, predictions, label='Predictions', color='green')

    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level')
    ax.set_title('Sea Level Prediction')

    plt.show()










layout = [
[sg.Button('Load data', size=(10,1), enable_events=True, key='-READ-', font='Helvetica 16'),
sg.Checkbox('Has column names?', size=(15,1), key='colnames-check',default=True),
sg.Checkbox('Drop NaN entries?', size=(15,1), key='drop-nan',default=True)],
[sg.Button('Show data', size=(10,1), enable_events=True, key='-SHOW-', font='Helvetica 16'),
sg.Button('Show stats', size=(15,1), enable_events=True, key='-STATS-', font='Helvetica 16')],
[sg.Text("", size=(50,1), key='-loaded-', pad=(5,5), font='Helvetica 14')],
[sg.Text("Select output column", size=(18,1), pad=(5,5), font='Helvetica 12')],
[sg.Listbox(values=(''), key='colnames', size=(30,3), enable_events=True)],
[sg.Text("", size=(50,1), key='-prediction-', pad=(5,5), font='Helvetica 12')],
[sg.Text("Select algorithm", size=(18,1), pad=(5,5), font='Helvetica 12')],
[sg.Combo(['Linear Regression', 'Random Forest', 'Decision Tree','KNeighbors Regressor','lasso Regression','Ridge Regression'], size=(30,1), key='algorithm')],
[sg.Button('Predict', size=(10,1), enable_events=True, key='-PREDICT-', font='Helvetica 16')],
[sg.ProgressBar(50, orientation='h', size=(100,20), key='progressbar')]
]

window = sg.Window('Sea Level Prediction', layout, size=(600,400))
progress_bar = window['progressbar']
prediction_text = window['-prediction-']
colnames_checked = False
dropnan_checked = False
read_successful = False

while True:
    event, values = window.read()
    loaded_text = window['-loaded-']
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == '-READ-':
        if values['colnames-check'] == True:
            colnames_checked = True
        if values['drop-nan'] == True:
            dropnan_checked = True
        try:
            df, data, header_list, fn = read_table()
            read_successful = True
        except:
            pass
        if read_successful:
            loaded_text.update("Dataset loaded: '{}'".format(fn))
            col_vals = [i for i in df.columns]            
            window.Element('colnames').Update(values=col_vals)
            # Split data into training and testing sets
            output_var = 'GMSL_noGIA'  # Replace with the appropriate output variable column name
            X_train, X_test, y_train, y_test = train_test_split(df.drop([output_var], axis=1), df[output_var], test_size=0.3, random_state=0)

    if event == '-SHOW-':
        if read_successful:
            show_table(data, header_list, fn)
        else:
            loaded_text.update("No dataset was loaded")
    if event == '-STATS-':
        if read_successful:
            show_stats(df)
        else:
            loaded_text.update("No dataset was loaded")
    if event == 'colnames':
        if len(values['colnames']) != 0:
            output_var = values['colnames'][0]
            if output_var != 'GMSL_noGIA':
                sg.Popup("Wrong output column selected!", title='Wrong', font="Helvetica 14")
            else:
                prediction_text.update("Fitting model...")
                for i in range(50):
                    event, values = window.read(timeout=10)
                    progress_bar.UpdateBar(i + 1)
            algorithm = values['algorithm']
            model, predictions_train, predictions_test, accuracy_train, accuracy_test = sklearn_model(output_var, algorithm, X_train, y_train, X_test, y_test)
            if model is not None:
                fig = plot_predictions(predictions_test, y_test)
                plt.show(block=False)
                window2 = sg.Window('Sea Level Prediction Plot', [[sg.Canvas(key='-CANVAS-')]], finalize=True)
                canvas = FigureCanvasTkAgg(fig, window2['-CANVAS-'].TKCanvas)
                canvas.draw()
                canvas.get_tk_widget().pack()
                window2.read()
                window2.close()
            else:
                sg.Popup("Invalid algorithm selected!", title='Error', font="Helvetica 14")
            prediction_text.update("r2 score of {} model is: {}".format(algorithm, accuracy_test))
            progress_bar.UpdateBar(0)

window.close()



    
    