import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from datetime import datetime
import csv
import platform
import pytz

# 在文件顶部添加平台检测
is_windows = platform.system() == 'Windows'

# 动态获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model", "catboost_model.cbm")
explainer_path = os.path.join(base_dir, "model", "explainer.shap")

# 根据操作系统选择不同的历史文件路径
if is_windows:
    history_dir = os.path.join(base_dir, "history")
    history_path = os.path.join(history_dir, "prediction_history.csv")
else:
    # 在Streamlit Cloud上使用/tmp目录确保有写入权限
    history_dir = "/tmp/history"
    history_path = os.path.join(history_dir, "prediction_history.csv")

# 确保历史记录目录存在
os.makedirs(history_dir, exist_ok=True)

# 初始化会话状态
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # 默认英语

# 多语言支持
translations = {
    'en': {
        'title': 'DR Risk Assessment System',
        'guest_access': 'Guest Access',
        'investigator_login': 'Investigator Login',
        'patient_info': 'Patient Information',
        'name': 'Name',
        'gender': 'Gender',
        'male': 'Male',
        'female': 'Female',
        'other': 'Other',
        'clinical_indicators': 'Enter Clinical Indicators',
        'start_assessment': 'Start Assessment',
        'warning_name': 'Please enter patient name before assessment.',
        'risk_probability': 'DR Risk Probability',
        'record_saved': 'Prediction record saved.',
        'feature_impact': 'Feature Impact Analysis',
        'feature_contribution': 'Feature Contribution Ranking',
        'prediction_history': 'Prediction History',
        'filter_by_name': 'Filter by patient name',
        'all': 'All',
        'delete_records': 'Delete Records',
        'select_records': 'Select records to delete (by index):',
        'delete_selected': 'Delete Selected Records',
        'download_history': 'Download History as CSV',
        'no_history': 'No prediction history yet.',
        'logout': 'Logout',
        'logged_in_as': 'Logged in as',
        'guest': 'guest',
        'investigator': 'investigator',
        'data_management': 'Data Management',
        'login_prompt': 'Please log in as an investigator to access this feature.',
        'username': 'Username',
        'password': 'Password',
        'enter_as_guest': 'Enter as Guest',
        'login_as_investigator': 'Login as Investigator',
        'invalid_credentials': 'Invalid username or password',
        'low_risk': 'Low Risk',
        'medium_risk': 'Medium Risk',
        'high_risk': 'High Risk',
        'risk_level': 'Risk Level'
    },
    'zh': {
        'title': '糖尿病视网膜病变风险评估系统',
        'guest_access': '访客入口',
        'investigator_login': '调查人员登录',
        'patient_info': '患者信息',
        'name': '姓名',
        'gender': '性别',
        'male': '男',
        'female': '女',
        'other': '其他',
        'clinical_indicators': '输入临床指标',
        'start_assessment': '开始评估',
        'warning_name': '评估前请输入患者姓名',
        'risk_probability': '糖尿病视网膜病变风险概率',
        'record_saved': '预测记录已保存',
        'feature_impact': '特征影响分析',
        'feature_contribution': '特征贡献度排名',
        'prediction_history': '预测历史',
        'filter_by_name': '按患者姓名筛选',
        'all': '全部',
        'delete_records': '删除记录',
        'select_records': '选择要删除的记录（按索引）:',
        'delete_selected': '删除选中的记录',
        'download_history': '下载历史记录',
        'no_history': '暂无预测历史',
        'logout': '退出登录',
        'logged_in_as': '登录身份',
        'guest': '访客',
        'investigator': '调查人员',
        'data_management': '数据管理',
        'login_prompt': '请以调查人员身份登录以访问此功能',
        'username': '用户名',
        'password': '密码',
        'enter_as_guest': '以访客身份进入',
        'login_as_investigator': '以调查人员身份登录',
        'invalid_credentials': '用户名或密码错误',
        'low_risk': '低风险',
        'medium_risk': '中风险',
        'high_risk': '高风险',
        'risk_level': '风险等级'
    }
}


def tr(key):
    """翻译函数"""
    return translations[st.session_state.language].get(key, key)


# 加载模型
try:
    model = CatBoostClassifier().load_model(model_path)
    explainer = joblib.load(explainer_path)
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()


# 初始化历史记录文件
def init_history_file():
    try:
        if not os.path.exists(history_path):
            features = ["Cortisol", "CRP", "Duration", "CysC", "C-P2", "BUN", "APTT", "RBG", "FT3", "ACR"]
            header = ["Timestamp", "Name", "Gender"] + features + ["Risk_Probability"]
            with open(history_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            # 设置文件权限 (Unix系统)
            if not is_windows:
                os.chmod(history_path, 0o644)
            return True
        return True
    except Exception as e:
        st.sidebar.error(f"初始化历史文件失败: {str(e)}")
        return False


# 保存预测记录到CSV
def save_prediction_record(name, gender, inputs, risk_probability):
    try:
        features = ["Cortisol", "CRP", "Duration", "CysC", "C-P2", "BUN", "APTT", "RBG", "FT3", "ACR"]
        # 获取北京时间
        beijing_tz = pytz.timezone('Asia/Shanghai')
        # 获取当前时间，并转换为北京时间
        beijing_time = datetime.now(beijing_tz)
        timestamp = beijing_time.strftime("%Y-%m-%d %H:%M:%S")

        record = [timestamp, name, gender] + [inputs[feat] for feat in features] + [risk_probability]

        # 检查文件是否存在，如果不存在则先创建
        if not os.path.exists(history_path):
            init_history_file()

        with open(history_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(record)

        return True
    except Exception as e:
        st.sidebar.error(f"保存记录失败: {str(e)}")
        return False


# 读取历史记录
def load_history():
    try:
        if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
            # 尝试不同编码
            try:
                history_df = pd.read_csv(history_path)
            except UnicodeDecodeError:
                history_df = pd.read_csv(history_path, encoding='latin-1')

            # 检查必要的列是否存在
            required_columns = ["Timestamp", "Name", "Risk_Probability"]
            if all(col in history_df.columns for col in required_columns):
                return history_df
            else:
                st.sidebar.warning("历史文件格式不正确，缺少必要列")
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        st.sidebar.error(f"读取历史文件时出错: {str(e)}")
        return pd.DataFrame()


# 删除选定的记录
def delete_records(records_to_delete):
    if os.path.exists(history_path):
        try:
            history_df = pd.read_csv(history_path)
            # 保留不在删除列表中的记录
            updated_history = history_df[~history_df.index.isin(records_to_delete)]
            # 保存更新后的记录
            updated_history.to_csv(history_path, index=False)
            return True
        except Exception as e:
            st.sidebar.error(f"删除记录时出错: {str(e)}")
            return False
    return False


# 初始化历史记录文件
init_history_file()

# 登录页面
if not st.session_state.logged_in:
    # 使用CSS添加自定义样式
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 30px;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 50px;
        }
        .title {
            text-align: center;
            color: #1f4e79;
            margin-bottom: 30px;
            font-size: 28px;
            font-weight: bold;
        }
        .eye-icon {
            font-size: 50px;
            text-align: center;
            margin-bottom: 20px;
        }
        .login-btn {
            width: 100%;
            margin-top: 20px;
        }
        .tab-title {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            border: 2px solid #1f4e79;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .unit-label {
            font-size: 12px;
            color: #666;
            margin-top: -10px;
            margin-bottom: 15px;
        }

        /* 完全移除所有选项卡的下划线 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            justify-content: center;
            border-bottom: none !important;
        }

        /* 移除选项卡激活指示器 */
        .stTabs [data-baseweb="tab-list"] > * {
            border-bottom: none !important;
        }

        /* 移除单个选项卡的底部边框 */
        .stTabs [data-baseweb="tab"] {
            height: 80px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 15px 25px;
            font-weight: bold;
            font-size: 18px;
            min-width: 180px;
            text-align: center;
            transition: all 0.3s ease;
            border: 2px solid transparent !important;
            box-shadow: none !important;
        }

        /* 移除悬停时的底部边框 */
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e6e9ed;
            transform: translateY(-2px);
            border: 2px solid #d0d5dd !important;
            box-shadow: none !important;
        }

        /* 移除选中选项卡的底部边框 */
        .stTabs [aria-selected="true"] {
            background-color: #1f4e79;
            color: white;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
            border: 2px solid #1f4e79 !important;
        }

        /* 移除选项卡高亮条 */
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # 语言选择器
    lang_col, _ = st.columns([0.2, 0.8])
    with lang_col:
        language = st.selectbox("", ["English", "中文"], index=0 if st.session_state.language == 'en' else 1,
                                key='lang_select')
        st.session_state.language = 'en' if language == "English" else 'zh'

    # 登录界面布局
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="eye-icon">👁️</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="title">{tr("title")}</h1>', unsafe_allow_html=True)

    # 创建两个选项卡，使用翻译后的文本作为标签
    tab1, tab2 = st.tabs([tr("guest_access"), tr("investigator_login")])

    with tab1:
        st.info(tr("guest_access") + " " + tr(
            "login_prompt") if st.session_state.language == 'en' else "访客用户只能进行风险评估，无法访问历史数据。")
        if st.button(tr("enter_as_guest"), key="guest_btn", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.user_type = "guest"
            st.rerun()

    with tab2:
        st.info(tr("investigator_login") + " " + tr(
            "login_prompt") if st.session_state.language == 'en' else "调查人员可以访问所有功能，包括数据管理。")
        username = st.text_input(tr("username"), key="username")
        password = st.text_input(tr("password"), type="password", key="password")

        if st.button(tr("login_as_investigator"), key="investigator_btn", use_container_width=True):
            if username == "DR" and password == "000000":
                st.session_state.logged_in = True
                st.session_state.user_type = "investigator"
                st.rerun()
            else:
                st.error(tr("invalid_credentials"))

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# 主应用界面
st.set_page_config(page_title="DR Prediction", layout="wide")

# 响应式设计 - 添加自定义CSS
st.markdown("""
    <style>
    @media (max-width: 768px) {
        .sidebar .sidebar-content {
            width: 100% !important;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
    }
    .unit-tooltip {
        font-size: 12px;
        color: #666;
        margin-top: -10px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 顶部状态栏显示当前用户类型和语言选择
col1, col2, col3 = st.columns([6, 1, 1])
with col1:
    st.title(f"👁️ {tr('title')}")
with col2:
    language = st.selectbox("", ["English", "中文"],
                            index=0 if st.session_state.language == 'en' else 1,
                            key='main_lang_select',
                            label_visibility="collapsed")
    st.session_state.language = 'en' if language == "English" else 'zh'
with col3:
    if st.button(tr("logout")):
        st.session_state.logged_in = False
        st.session_state.user_type = None
        st.rerun()
    st.caption(f"{tr('logged_in_as')}: {tr(st.session_state.user_type)}")

# 用户信息输入
with st.sidebar:
    st.header(tr("patient_info"))
    name = st.text_input(tr("name"))
    gender = st.selectbox(tr("gender"), [tr("male"), tr("female"), tr("other")])

    st.header(tr("clinical_indicators"))
    inputs = {}
    features = ["Cortisol", "CRP", "Duration", "CysC", "C-P2", "BUN", "APTT", "RBG", "FT3", "ACR"]
    units = {
        "Cortisol": "μg/L",
        "CRP": "mg/L",
        "Duration": "year",
        "CysC": "mg/L",
        "C-P2": "ng/ml",
        "BUN": "mmol/L",
        "APTT": "s",
        "RBG": "mmol/L",
        "FT3": "pmol/L",
        "ACR": "Urine Protein/Creatinine Ratio"
    }

    # 添加自定义CSS以允许清空数字输入框
    st.markdown("""
    <style>
    /* 允许数字输入框被清空 */
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    input[type="number"] {
        -moz-appearance: textfield;
    }
    </style>
    """, unsafe_allow_html=True)

    for feat in features:
        # 使用数字输入框，但添加一个清除按钮
        col1, col2 = st.columns([4, 1])
        with col1:
            # 使用session_state来跟踪每个输入框的值
            if f"input_{feat}" not in st.session_state:
                st.session_state[f"input_{feat}"] = 0.0

            inputs[feat] = st.number_input(
                feat,
                value=st.session_state[f"input_{feat}"],
                key=f"num_input_{feat}",
                format="%.2f"
            )
            # 更新session_state中的值
            st.session_state[f"input_{feat}"] = inputs[feat]

        with col2:
            # 添加清除按钮
            if st.button("×", key=f"clear_{feat}"):
                st.session_state[f"input_{feat}"] = 0.0
                st.rerun()

        st.markdown(f'<div class="unit-tooltip">{units[feat]}</div>', unsafe_allow_html=True)
# 预测与解释
if st.button(tr("start_assessment"), type="primary"):
    if not name:
        st.warning(tr("warning_name"))
    else:
        input_df = pd.DataFrame([inputs])

        # 预测概率
        prob = model.predict_proba(input_df)[0][1]

        # 使用颜色编码显示风险水平
        if prob < 0.3:
            risk_color = "green"
            risk_level = tr("low_risk")
        elif prob < 0.7:
            risk_color = "orange"
            risk_level = tr("medium_risk")
        else:
            risk_color = "red"
            risk_level = tr("high_risk")

        st.markdown(f"""
        <div style="padding: 15px; border-radius: 5px; background-color: #f8f9fa; border-left: 5px solid {risk_color}; margin-bottom: 20px;">
            <h3 style="color: {risk_color}; margin-top: 0;">{tr('risk_probability')}: <b>{prob * 100:.1f}%</b></h3>
            <p>{tr('risk_level')}: <b>{risk_level}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # 保存预测记录
        if save_prediction_record(name, gender, inputs, prob):
            st.info(tr("record_saved"))
        else:
            st.error("保存记录失败，请查看侧边栏的错误信息")

        # SHAP解释
        st.subheader(tr("feature_impact"))
        shap_value = explainer.shap_values(input_df)

        # 绘制Force Plot
        plt.figure(figsize=(10, 4))
        shap.force_plot(
            explainer.expected_value,
            shap_value[0],
            input_df.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(plt.gcf())

        # 特征重要性表格
        st.subheader(tr("feature_contribution"))
        contribution_df = pd.DataFrame({
            "Feature": features,
            "SHAP Value": shap_value[0]
        }).sort_values("SHAP Value", key=abs, ascending=False)


        # 添加颜色编码
        def color_shap_value(val):
            color = 'red' if val > 0 else 'green'
            return f'color: {color}'


        styled_df = contribution_df.style.applymap(color_shap_value, subset=['SHAP Value'])
        st.dataframe(styled_df)

# 历史记录查询 - 仅对调查人员开放
if st.session_state.user_type == "investigator":
    st.subheader(tr("prediction_history"))
    history_df = load_history()

    # 添加调试按钮
    if st.sidebar.button("调试历史记录"):
        st.sidebar.write(f"历史文件路径: {history_path}")
        st.sidebar.write(f"文件存在: {os.path.exists(history_path)}")
        if os.path.exists(history_path):
            st.sidebar.write(f"文件大小: {os.path.getsize(history_path)} 字节")
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                st.sidebar.text_area("文件内容", content, height=200)
            except Exception as e:
                st.sidebar.error(f"读取文件内容失败: {str(e)}")

    if not history_df.empty:
        # 添加索引列以便选择
        history_df_display = history_df.copy()
        history_df_display['Index'] = history_df_display.index

        # 创建顶部控制行 - 筛选和下载按钮在同一行
        control_col1, control_col2 = st.columns([3, 1])

        with control_col1:
            # 按姓名筛选
            names = history_df['Name'].unique()
            selected_name = st.selectbox(tr("filter_by_name"), [tr("all")] + list(names))

        with control_col2:
            # 提供下载选项 - 放在筛选框同一行
            csv = history_df_display.drop(columns=['Index']).to_csv(index=False)
            st.download_button(
                label=tr("download_history"),
                data=csv,
                file_name="dr_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        if selected_name != tr("all"):
            filtered_history = history_df_display[history_df_display['Name'] == selected_name]
        else:
            filtered_history = history_df_display

        # 显示历史记录
        st.dataframe(filtered_history.sort_values("Timestamp", ascending=False).drop(columns=['Index']))

        # 删除记录功能
        st.subheader(tr("data_management"))

        # 选择要删除的记录
        records_to_delete = st.multiselect(
            tr("select_records"),
            options=filtered_history['Index'].tolist(),
            format_func=lambda
                x: f"Index {x}: {filtered_history[filtered_history['Index'] == x]['Name'].iloc[0]} - {filtered_history[filtered_history['Index'] == x]['Timestamp'].iloc[0]}"
        )

        if records_to_delete and st.button(tr("delete_selected"), type="secondary"):
            if delete_records(records_to_delete):
                st.success("Selected records deleted successfully!")
                # 重新加载页面以更新显示
                st.rerun()
            else:
                st.error("Failed to delete records.")
    else:
        st.info(tr("no_history"))
        # 提供创建示例数据的选项
        if st.button("创建示例数据"):
            sample_data = {
                "Timestamp": [datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")],
                "Name": ["示例患者"],
                "Gender": ["Male"],
                "Cortisol": [15.2],
                "CRP": [3.5],
                "Duration": [5],
                "CysC": [1.2],
                "C-P2": [2.8],
                "BUN": [6.5],
                "APTT": [32],
                "RBG": [7.2],
                "FT3": [4.5],
                "ACR": [30],
                "Risk_Probability": [0.45]
            }
            sample_df = pd.DataFrame(sample_data)
            sample_df.to_csv(history_path, index=False)
            st.rerun()
else:
    st.info(tr("login_prompt"))