import pandas as pd
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, scrolledtext
import tksheet
from tkhtmlview import HTMLLabel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import markdown
from io import StringIO
import subprocess
from scipy import stats
import numpy as np
from statsmodels.stats.stattools import jarque_bera

# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ =========================================================================================================
pd.set_option('display.max_rows', None)

current_file = None
current_data = None

# ФУНКЦИИ ========================================================================================================================

# функция смены панели инструментов по вкладкам
def switch_toolbar(event):
    # Получаем индекс выбранной вкладки
    selected_tab = notebook.index(notebook.get())
    # Удаляем все виджеты с текущей панели инструментов
    for widget in add_toolbar.winfo_children():
        widget.pack_forget()

    # Для каждой вкладки отображаем соответствующую панель инструментов
    if selected_tab == 0:  # Вкладка "Таблица"
        btn_open_csv.pack(side = 'left', padx = 5, pady = 5)
        btn_information.pack(side = 'left', padx = 5, pady = 5)
        btn_deep_analysis.pack(side = 'left', padx = 5, pady = 5)
    elif selected_tab == 1:  # Вкладка "Диаграмма"
        btn_save_chart.pack(side ="left", padx = 5, pady = 5)
    elif selected_tab == 2:  # Вкладка "Машинное обучение"
        btn_new_model.pack(side ="left", padx = 5, pady = 5)
        btn_open_script.pack(side ="left", padx = 5, pady = 5)
        btn_run_script.pack(side ="left", padx = 5, pady = 5)
        btn_save_script.pack(side = 'left', padx = 5, pady=5)
        btn_save_as.pack(side ="left", padx = 5, pady = 5)
        btn_help_ml.pack(side = "left", padx = 5, pady = 5)

# функция открытия файла через окно
def open_file():
    global current_file, current_data
    
    filepath = filedialog.askopenfilename(filetypes = [("CSV Files", "*.csv")])
    if filepath:
        current_file = filepath
        current_data = pd.read_csv(filepath)
        show_table(current_data)
        update_chart_panel()


# ВКЛАДКА "ТАБЛИЦА" ============================================================================================================

# функция отображения файла в табличном виде
def show_table(data):
    global sheet, headers
    
    # Очищаем содержимое фреймов
    for widget in table_frame_head.winfo_children():
        widget.destroy()
    
    for widget in table_frame_tail.winfo_children():
        widget.destroy()
    
    # Используем текущие данные вместо несуществующего current_data
    table_data_head = data.head(15).values.tolist()
    table_data_tail = data.tail(15).values.tolist()
    
    # Основной объект sheet для анализа данных
    sheet = data
    headers = list(data.columns)
    
    # Создание и отображение верхней части таблицы
    sheet_head = tksheet.Sheet(table_frame_head,
                               data = table_data_head,
                               headers = list(data.columns),
                               show_x_scrollbar = True,
                               show_y_scrollbar = True,
                               table_bg='#EFEFEF')
    sheet_head.pack(expand=True, fill='both')

    # Создание и отображение нижней части таблицы
    sheet_tail = tksheet.Sheet(table_frame_tail,
                               data = table_data_tail,
                               headers = list(data.columns),
                               show_x_scrollbar = True,
                               show_y_scrollbar = True,
                               table_bg='#EFEFEF')
    sheet_tail.pack(expand=True, fill='both')

    # Включение необходимых событий
    sheet_head.enable_bindings(("single_select",
                                "column_select",
                                "row_select",
                                "arrowkeys",
                                "edit_cell"))
    
    sheet_tail.enable_bindings(("single_select",
                                "column_select",
                                "row_select",
                                "arrowkeys",
                                "edit_cell"))

    # Привязка обновления графика при редактировании
    sheet_head.extra_bindings([("end_edit_cell", update_chart)])
    sheet_tail.extra_bindings([("end_edit_cell", update_chart)])
    
# Функция сохранения
def save_changes():
    save_filepath = filedialog.asksaveasfilename(defaultextension=".csv", 
                                                filetypes=[("CSV files", "*.csv")])
    
    if save_filepath:
        updated_data = pd.DataFrame(sheet.get_sheet_data(), columns=sheet.headers())
        updated_data.to_csv(save_filepath, index=False)

def show_info(info):
    # Очищаем текстовое поле и вставляем новую информацию
    text_box.delete('1.0', tk.END)
    text_box.insert(tk.END, info)

def create_analysis_window():
    # Создаем новое окно
    analysis_window = ctk.CTkToplevel()
    analysis_window.title("Анализ данных")
    analysis_window.geometry('1000x400')
    analysis_window.grab_set()
    
    # Создаем фрейм для дерева и текстового поля
    frame = ctk.CTkFrame(analysis_window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Создаем дерево
    tree = ttk.Treeview(frame)
    tree.pack(side=tk.LEFT, fill=tk.Y)

    # Создаем фрейм для текстового поля и скроллов
    text_frame = ctk.CTkFrame(frame)
    text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    global process_frame
    process_frame = ctk.CTkFrame(frame, width = 35)
    process_frame.pack_propagate(False)
    process_frame.pack(side = tk.RIGHT, fill = tk.BOTH, expand = True)

    # Создаем текстовое поле для отображения информации
    global text_box
    text_box = tk.Text(text_frame, wrap=tk.NONE, width=60, height=20)
    
    # Создаем вертикальный скроллбар
    scrollbar_Y = ctk.CTkScrollbar(text_frame, command=text_box.yview, width=15)
    scrollbar_Y.pack(side=tk.RIGHT, fill=tk.Y)
    text_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Создаем горизонтальный скроллбар
    scrollbar_X = ctk.CTkScrollbar(text_frame, command=text_box.xview, height=15, orientation=tk.HORIZONTAL)
    scrollbar_X.pack(side=tk.BOTTOM, fill=tk.X)

    # Настраиваем текстовое поле для работы с прокрутками
    text_box.configure(yscrollcommand=scrollbar_Y.set, xscrollcommand=scrollbar_X.set)
    
    # Заполнение дерева
    tree.insert("", "end", "info", text="Общая информация")
    tree.insert("", "end", "stats", text="Статистическое описание")
    tree.insert("", "end", "missing", text="Проверка пропусков")
    tree.insert("", "end", "unique", text="Уникальные значения")
    tree.insert("", "end", "duplicates", text="Дубликаты")

    # Привязываем события для выбора пункта
    tree.bind("<<TreeviewSelect>>", lambda e: on_tree_select(tree) and create_processing_options(tree))

def on_tree_select(tree):
    global current_data  # Убедитесь, что current_data загружен
    global selected_item
    selected_item = tree.selection()[0]
    if selected_item == "info":
        # Общая информация о данных
        create_processing_options()
        buffer = StringIO()
        current_data.info(buf=buffer)
        info = buffer.getvalue()
    elif selected_item == "stats":
        # Статистическое описание
        create_processing_options()
        info = current_data.describe().to_string()
    elif selected_item == "missing":
        # Проверка пропусков
        create_processing_options()
        missing_info = current_data.isnull().sum()
        info = f"Проверка пропусков:\n{missing_info[missing_info > 0]}"
    elif selected_item == "unique":
        # Уникальные значения
        create_processing_options()
        unique_info = current_data.nunique()
        info = f"Уникальные значения:\n{unique_info}"
    elif selected_item == "duplicates":
        # Дубликаты
        create_processing_options()
        duplicate_count = current_data.duplicated().sum()
        info = f"Количество дубликатов: {duplicate_count}"

    show_info(info)
    update_chart_panel()

def create_processing_options():
    for widget in process_frame.winfo_children():
        widget.destroy()  # Очищаем фрейм перед добавлением новых виджетов
    
    if selected_item == "info":
        # Чистка по столбцам и нормализация
        columns = current_data.columns.to_list()
        col_var = tk.StringVar(value=columns[0])  # Выбор одного столбца
        
        comboBox = ctk.CTkComboBox(process_frame, values=columns, variable=col_var)
        comboBox.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)       
        
        # Добавляем кнопку "Применить"
        apply_button = ctk.CTkButton(process_frame, text="Применить", command=lambda: apply_processing(selected_item, col_var, method_var))
        apply_button.pack(side=tk.BOTTOM, pady=10)
        
        # Методы обработки
        method_var = tk.StringVar(value="Удалить")
        methods = ["Удалить"]
        
        for method in methods:
            ctk.CTkRadioButton(process_frame, text=method, variable=method_var, value=method).pack(anchor=tk.W, padx=10, pady=5)
    
    if selected_item == "missing":
        # Чек-лист столбцов
        columns = current_data.columns.to_list()
        col_var = tk.StringVar(value=columns[0])  # Выбор одного столбца
        
        comboBox = ctk.CTkComboBox(process_frame, values=columns, variable=col_var)
        comboBox.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Добавляем кнопку "Применить"
        apply_button = ctk.CTkButton(process_frame, text="Применить", command=lambda: apply_processing(selected_item, col_var, method_var))
        apply_button.pack(side=tk.BOTTOM, pady=10)
        
        # Методы обработки
        method_var = tk.StringVar(value="Удалить")
        methods = ["Удалить", "Среднее", "Медиана"]
        
        for method in methods:
            ctk.CTkRadioButton(process_frame, text=method, variable=method_var, value=method).pack(anchor=tk.W, padx=10, pady=5)

    elif selected_item == "duplicates":
        # Чек-лист столбцов для дубликатов
        columns = current_data.columns.to_list()
        col_var = tk.StringVar(value=columns[0])  # Выбор одного столбца
        
        comboBox = ctk.CTkComboBox(process_frame, values=columns, variable=col_var)
        comboBox.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Добавляем кнопку "Применить"
        apply_button = ctk.CTkButton(process_frame, text="Применить", command=lambda: apply_processing(selected_item, col_var, method_var))
        apply_button.pack(side=tk.BOTTOM, pady=10)
        
        # Методы обработки дубликатов
        method_var = tk.StringVar(value="Удалить")
        methods = ["Удалить", "Оставить уникальные"]
        
        for method in methods:
            ctk.CTkRadioButton(process_frame, text=method, variable=method_var, value=method).pack(anchor=tk.W, padx=10, pady=5)
    
def apply_processing(selected_item, col_var, method_var):
    selected_col = col_var.get()  # Получаем выбранный столбец
    method = method_var.get()  # Получаем выбранный метод
    
    if selected_item == "info":
        
        if method == "Удалить":
            current_data.drop([selected_col], axis = 1, inplace = True)
    
    if selected_item == "missing":
        # Обработка пропусков
        if method == "Удалить":
            current_data.dropna(subset=[selected_col], inplace = True)
        elif method == "Среднее":
            current_data[selected_col].fillna(current_data[selected_col].mean(), inplace = True)
        elif method == "Медиана":
            current_data[selected_col].fillna(current_data[selected_col].median(), inplace = True)

    elif selected_item == "duplicates":
        # Обработка дубликатов
        if method == "Удалить":
            current_data.drop_duplicates(subset=[selected_col], inplace = True)
        elif method == "Оставить уникальные":
            current_data.drop_duplicates(subset=[selected_col], keep='first', inplace = True)
    
    # Обновление таблицы
    show_table(current_data)

# Функция для глубокого анализа данных и генерации отчета
def deep_analysis():
    data = sheet
    file_name = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("TXT", "*.txt")])
    report = []

    # Общая информация
    report.append(f"Общая информация о данных:\n")
    buffer = StringIO()
    data.info(buf=buffer)
    report.append(buffer.getvalue())

    # Статистическое описание
    report.append(f"\nСтатистическое описание:\n")
    report.append(data.describe().to_string())

    # Пропущенные значения
    missing = data.isnull().sum()
    if missing.any():
        report.append(f"\nПропущенные значения:\n{missing[missing > 0]}")
    else:
        report.append("\nПропущенные значения: отсутствуют")

    # Корреляционный анализ
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        correlation_matrix = data[numeric_columns].corr()
        report.append(f"\nКорреляционный анализ:\n{correlation_matrix.to_string()}")
    else:
        report.append("\nНедостаточно числовых данных для корреляционного анализа")

    # Анализ выбросов (метод IQR)
    report.append("\nАнализ выбросов:\n")
    for column in numeric_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))]
        if not outliers.empty:
            report.append(f"Выбросы в столбце {column}:\n{outliers}")
        else:
            report.append(f"Выбросы в столбце {column}: отсутствуют")

    # Статистические тесты
    report.append("\nСтатистические тесты:\n")
    for column in numeric_columns:
        col_data = data[column].dropna()

        # Тест Шапиро-Уилка на нормальность
        stat, p_value = stats.shapiro(col_data)
        report.append(f"Тест Шапиро-Уилка для {column}: p-value = {p_value:.5f}")
        
        # Тест Колмогорова-Смирнова на нормальность
        stat, p_value = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
        report.append(f"Тест Колмогорова-Смирнова для {column}: p-value = {p_value:.5f}")

        # Тест Левена для проверки равенства дисперсий
        if len(numeric_columns) > 1:
            stat, p_value = stats.levene(*[data[col].dropna() for col in numeric_columns])
            report.append(f"Тест Левена на равенство дисперсий для {column}: p-value = {p_value:.5f}")
        
        # Исправленная распаковка для Jarque-Bera
        jb_stat, jb_pvalue, _, _ = jarque_bera(col_data)
        report.append(f"Тест Jarque-Bera для {column}: p-value = {jb_pvalue:.5f}")

        # Z-score для выбросов
        z_scores = np.abs(stats.zscore(col_data))
        outliers_z = col_data[z_scores > 3]
        if not outliers_z.empty:
            report.append(f"Выбросы по Z-оценке в {column}:\n{outliers_z}")
        else:
            report.append(f"Выбросы по Z-оценке в {column}: отсутствуют")

    # Сохранение отчета в текстовый файл
    with open(file_name, 'w') as file:
        file.write("\n".join(report))

    return file_name

# ВКЛАДКА "ДИАГРАММА" ==================================================================================================================

# Функция для отображения графика
def show_chart():
    global current_data
    if current_data is not None:
        update_chart()

# Функция для отображения и обновления диаграммы на основе выбранных столбцов
def update_chart():
    selected_columns = [var.get() for var in column_vars if var.get() in headers]
    
    if selected_columns:
        updated_data = pd.DataFrame(sheet, columns=headers)
        ax.clear()
        updated_data[selected_columns].plot(ax=ax)
        canvas.draw()

# Вкладка "Диаграмма"
def update_chart_panel():
    global column_vars
    global chart_type, title, x_col, y_col, hue
    
    if current_data is None:
        return
    
    for widget in visual_frame.winfo_children():
        widget.destroy()
    
     # Создаем область для чекбоксов с прокруткой
    column_listbox_frame = ctk.CTkFrame(visual_frame)
    column_listbox_frame.pack(side='left', fill='y', padx=5, pady=5)
    
    chart_settings = ctk.CTkButton(column_listbox_frame, text='Настроить')
    chart_settings.pack(side='top', fill='y', padx=5, pady=5)
    
    # Создаем холст для фрейма с прокруткой
    scrollFrame = tk.Canvas(column_listbox_frame)
    scrollFrame.pack_propagate(False)
    scrollbar = ctk.CTkScrollbar(column_listbox_frame, orientation="vertical", command=scrollFrame.yview)
    scrollable_frame = ctk.CTkFrame(scrollFrame)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: scrollFrame.configure(
            scrollregion=scrollFrame.bbox("all")
        )
    )
    
    scrollFrame.create_window((0, 0), window=scrollable_frame, anchor="nw")
    scrollFrame.configure(yscrollcommand=scrollbar.set)
    
    # Размещаем холст и скроллбар
    scrollFrame.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    column_vars = []
    
    # Обновляем чекбоксы для выбора столбцов внутри прокручиваемого фрейма
    for column in headers:
        var = tk.StringVar(value='')
        checkbutton = ctk.CTkCheckBox(scrollable_frame, text=column, variable=var, onvalue=column, offvalue='')
        checkbutton.pack(anchor='w', pady=5)
        column_vars.append(var)
        checkbutton.configure(command=update_chart)
    
    
    # Поле для графика
    chart_frame = ttk.Frame(visual_frame)
    chart_frame.pack(side='right', fill='both', expand=True)
    
    global fig, ax, canvas
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.get_tk_widget().pack(fill='both', expand=True)

# Функция сохранения диаграммы
def save_chart():
    save_filepath = filedialog.asksaveasfilename(defaultextension=".png", 
                                                filetypes=[("PNG files", "*.png")])
    
    if save_filepath:
        fig.savefig(save_filepath)

# ВКЛАДКА МАШИННОЕ ОБУЧЕНИЕ==========================================================================================

def open_ml_editor():
    
    # Пример базового шаблона кода для моделей машинного обучения
    base_code = '''\
# Импортируем библиотеки
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Загружаем данные
data = pd.read_csv('your_dataset.csv')
X = data.drop('target_column', axis=1)
y = data['target_column']

# Разбиваем данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
'''

    # Вставляем базовый код в редактор
    code_editor.insert(tk.INSERT, base_code)

# Переменная для хранения пути к текущему открытому файлу
current_file_path = None

# Функция для загрузки скрипта в текстовый редактор
def load_code():
    global current_file_path
    # Открываем диалоговое окно для выбора файла
    current_file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
    
    if current_file_path:
        # Читаем файл и загружаем содержимое в текстовый редактор
        with open(current_file_path, 'r', encoding = 'utf-8') as file:
            code = file.read()
            code_editor.delete(1.0, tk.END)  # Очистить текстовый редактор
            code_editor.insert(tk.END, code)  # Загрузить код в редактор

# Функция для выполнения кода из редактора
def run_code():
    global current_file_path

    if current_file_path:  # Если файл уже сохранен
        with open(current_file_path, 'w', encoding='utf-8') as file:
            file.write(code_editor.get(1.0, tk.END))

    else:  # Если файл новый, запрашиваем место для сохранения
        current_file_path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python files", "*.py")])
        if not current_file_path:  # Если пользователь отменил сохранение
            print("Сохранение отменено. Код не запущен.")
            return
        with open(current_file_path, 'w', encoding='utf-8') as file:
            file.write(code_editor.get(1.0, tk.END))

    # Запускаем скрипт через subprocess и выводим результат в консоль
    process = subprocess.Popen(['python', current_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if stdout:
        print(f"Output:\n{stdout}")
    if stderr:
        print(f"Errors:\n{stderr}")

# Функция для сохранения кода
def save_code():
    global current_file_path

    if current_file_path:  # Если файл уже существует
        with open(current_file_path, 'w', encoding='utf-8') as file:
            file.write(code_editor.get(1.0, tk.END))
        print(f"Файл сохранен: {current_file_path}")

    else:  # Если файл новый, запрашиваем место для сохранения
        current_file_path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python files", "*.py")])
        if current_file_path:  # Если пользователь выбрал место для сохранения
            with open(current_file_path, 'w', encoding='utf-8') as file:
                file.write(code_editor.get(1.0, tk.END))
            print(f"Файл сохранен: {current_file_path}")
        else:
            print("Сохранение отменено.")
        
def save_as():
    global current_file_path
    # Открываем диалог для сохранения файла
    current_file_path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python files", "*.py")])
    if current_file_path:
        # Сохраняем содержимое редактора в выбранный файл
        with open(current_file_path, 'w', encoding = 'utf-8') as file:
            file.write(code_editor.get(1.0, tk.END))

# Функция для отображения содержимого выбранного файла
def display_file_content(file_path, content_frame):
    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
        html_content = markdown.markdown(md_content)  # Конвертируем в HTML
        
        # Очистка предыдущего содержимого
        for widget in content_frame.winfo_children():
            widget.destroy()
        
        # Отображаем HTML с помощью HTMLLabel
        html_label = HTMLLabel(content_frame, html=html_content)
        html_label.pack(fill="both", expand=True)

# Функция для создания окна с деревом каталогов
def open_ml_info_window():
    # Создание нового окна
    window = ctk.CTkToplevel()
    window.grab_set()  # Удерживает фокус на новом окне
    window.title("ML Info")
    window.geometry("800x600")

    # Создание дерева каталогов слева
    tree_frame = ctk.CTkFrame(window)
    tree_frame.pack(side="left", fill="y")

    tree = ttk.Treeview(tree_frame)
    tree.pack(fill="y", expand=True)

    # Фрейм для отображения содержимого файла справа
    content_frame = ctk.CTkFrame(window)
    content_frame.pack(side="right", fill="both", expand=True)

    # Функция для добавления каталогов и файлов в дерево
    def populate_tree(parent, path):
        for item in os.listdir(path):
            abs_path = os.path.join(path, item)
            if os.path.isdir(abs_path):
                node = tree.insert(parent, 'end', text=item, open=False)
                populate_tree(node, abs_path)
            elif item.endswith(".md"):
                tree.insert(parent, 'end', text=item[:-3], values=[abs_path])

    # Добавляем корневой каталог в дерево
    root_dir = "./info"
    populate_tree("", root_dir)

    # Привязка события выбора элемента в дереве
    def on_tree_select(event):
        selected_item = tree.selection()[0]
        file_path = tree.item(selected_item, "values")
        if file_path:
            display_file_content(file_path[0], content_frame)

    tree.bind("<<TreeviewSelect>>", on_tree_select)
    
# 1. Системная панель =====================================================================================================================

# Главное окно
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
root = ctk.CTk() 
root.title('VKR Project')
root.geometry('1270x600')

toolbar = tk.Menu(root)
root.config(menu = toolbar)

# 1.1. Кнопки и команды
file_menu = tk.Menu(toolbar, tearoff=0)
toolbar.add_cascade(label = "Файл", menu = file_menu)
file_menu.add_command(label = "Открыть CSV файл", command=open_file)
file_menu.add_command(label = "Сохранить как", command=save_changes)

# 2. Панель инструментов ======================================================================================================================

# панель инструментов
add_toolbar = ctk.CTkFrame(root, width=200, height=50, fg_color = 'transparent')
add_toolbar.pack(side = 'top', fill = 'x')

# Кнопки для вкладки "Таблица"
btn_open_csv = ctk.CTkButton(add_toolbar, text = 'Открыть', command = open_file)
btn_information = ctk.CTkButton(add_toolbar, text = 'Анализ', command = create_analysis_window)
btn_deep_analysis = ctk.CTkButton(add_toolbar, text = 'Глубокий анализ', command = deep_analysis)

# Кнопки для вкладки "Диаграмма"
btn_save_chart = ctk.CTkButton(add_toolbar, text = 'Сохранить', command = save_chart)

# Кнопки для вкладки "Машинное обучение"
btn_new_model = ctk.CTkButton(add_toolbar, text = 'Шаблон', command = open_ml_editor)
btn_open_script = ctk.CTkButton(add_toolbar, text = 'Загрузить', command = load_code)
btn_run_script = ctk.CTkButton(add_toolbar, text = 'Запуск', command = run_code)
btn_save_script = ctk.CTkButton(add_toolbar, text = "Сохранить", command = save_code)
btn_save_as = ctk.CTkButton(add_toolbar, text = 'Сохранить как', command = save_as)
btn_help_ml = ctk.CTkButton(add_toolbar, text = 'Помощь', command = open_ml_info_window)


# 3. Вкладки страниц =========================================================================================================================

notebook = ctk.CTkTabview(root)
notebook.pack(fill='both', expand=True)

notebook.add('Таблица')
notebook.add('Диаграмма')
notebook.add('Обучение')
notebook.add('Сбор')

table_frame_head = ctk.CTkFrame(notebook.tab('Таблица'))
table_frame_head.pack(fill='both', expand = True, pady = 5)

table_frame_tail = ctk.CTkFrame(notebook.tab('Таблица'))
table_frame_tail.pack(fill='both', expand = True, pady = 5)

visual_frame = ctk.CTkFrame(notebook.tab('Диаграмма'))
visual_frame.pack(fill = 'both', expand = True)

ml_frame = ctk.CTkFrame(notebook.tab('Обучение'))
ml_frame.pack(fill = 'both', expand = True)


global code_editor
code_editor = ctk.CTkTextbox(ml_frame)
code_editor.pack(expand = True, fill = 'both')

# ИНИЦИАЛИЗАЦИЯ (ну почти) ===================================================================================================================

def check_active_tab():
    switch_toolbar(None)
    root.after(100, check_active_tab)

root.after(100, check_active_tab)

def on_close():
    root.after_cancel(root.after(100, check_active_tab))  # Остановить все задачи after
    root.quit()

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
