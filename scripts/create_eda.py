"""Build notebook reports using measured values from the local data snapshot."""
from pathlib import Path
import sys
import textwrap

import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.eda_utils import (
    TASKS, load_task, quality_table, bounds_table, outlier_table,
    association, period_changes, strongest_pairs,
)


class Notebook:
    def __init__(self, title):
        self.cells = []
        self.md(title)

    def md(self, text):
        self.cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(self, text):
        self.cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    def notes(self, finding, interpretation, implication, action):
        self.md(f'**Findings & Interpretation**\n\n**Findings:** {finding}\n\n'
                f'**Interpretation:** {interpretation}\n\n**ML Implications:** {implication}\n\n'
                f'**Recommended Action:** {action}')

    def save(self, path):
        notebook = nbf.v4.new_notebook(cells=self.cells)
        notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
        path.parent.mkdir(exist_ok=True)
        nbf.write(notebook, path)
        print('Created', path.relative_to(ROOT), flush=True)


SETUP = """
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'sample_submission.csv').exists())
sys.path.insert(0, str(ROOT))
from shared.eda_utils import (
    TASKS, load_task, quality_table, outlier_table, bounds_table,
    association, strongest_pairs, period_changes, image_inventory, setup_plot,
)
setup_plot()
"""

TITLES = {1: 'Anti-cheat detection', 2: 'Player segmentation',
          3: 'Monthly spending', 4: 'Game title detection', 5: 'Account anomaly detection'}
OBJECTIVES = {
    1: 'ทำนายผู้เล่นที่อาจโกง เพื่อจัดลำดับการตรวจสอบและลด false positive ต่อผู้เล่นปกติ',
    2: 'จำแนกกลุ่มผู้เล่นเพื่อปรับกิจกรรมและข้อเสนอให้เหมาะกับพฤติกรรม',
    3: 'ทำนายยอดใช้จ่ายใน 30 วันถัดไป หน่วย THB เพื่อวางแผนรายได้และกิจกรรม',
    4: 'ระบุชื่อเกมจาก screenshot เพื่อจัดหมวดภาพและคำร้องของผู้เล่น',
    5: 'ค้นหาบัญชีที่มีพฤติกรรมเปลี่ยนไป เพื่อจัดลำดับการตรวจสอบความปลอดภัย',
}
CARDS = {
    1: '| กลุ่ม | ความหมายตาม data card |\n|---|---|\n| performance | kill/death, headshot, accuracy และ win rate |\n| timing/input | reaction time, input consistency, aiming และ movement |\n| account/social | อายุบัญชี level friend network reports และ session |\n| tactical | game sense, teamwork, map knowledge, utility และ crosshair |',
    2: '| กลุ่ม | ความหมายตาม data card |\n|---|---|\n| engagement | play frequency, session duration, playtime และ login |\n| spending | ยอดใช้จ่าย ความถี่ซื้อ VIP และ loyalty |\n| social/competitive | เพื่อน chat team play rank และ tournament |\n| categorical | region, platform, device, payment, language และ tier |\n| random_metric_1–3 | data card ระบุว่าเป็น noise |',
    3: '| กลุ่ม | ความหมายตาม data card |\n|---|---|\n| social/engagement | friend, guild, event, session และ achievement |\n| spending history | ยอดใช้จ่ายก่อนหน้า จำนวนรายการ และมูลค่ารายการ |\n| membership | VIP, premium และ segment |\n| purchase context | เวลาหลังซื้อครั้งล่าสุด ส่วนลด platform และ seasonal pattern |',
    4: '| ค่าใน CSV | รหัสคำตอบ | เกมตาม data card |\n|---|---|---|\n| 7-11k | 0 | 7-11 Knights |\n| fiveN | 1 | FiveN |\n| freefried | 2 | Free Fried |\n| pube | 3 | PUBE |\n| roblock | 4 | Roblock |\n\nภาพ JPEG คาดหวังขนาด 230×120, RGB 8-bit; id เชื่อมกับ `{split}/{id}.jpg` ภายในโฟลเดอร์ข้อมูลภาพจริง',
    5: '| กลุ่ม | ความหมายตาม data card |\n|---|---|\n| login/location/device | จำนวน login พิกัด การเปลี่ยนที่อยู่ อุปกรณ์ และ IP entropy |\n| session/activity | เวลา login ระยะ session pattern consistency และ activity |\n| performance | skill rating/trajectory, KDA, accuracy, reaction time และ stability |\n| purchase/social | ความถี่และยอดซื้อ refund transfer friend communication และ guild |\n| account/flags | อายุบัญชี stability password VPN เวลา login ผิดปกติและ mass sale |\n\nSuffix `_1` ถึง `_4` เป็น 4 ช่วงเวลาต่อเนื่อง ไม่ทราบความยาวแต่ละช่วง และแต่ละแถวคือบัญชีหนึ่งบัญชี',
}


def build_tabular(task):
    train, test, target, base = load_task(ROOT, task)
    frame = train if train is not None else test
    features = frame.drop(columns=['player_id', target], errors='ignore')
    missing = features.isna().mean().mean() * 100
    quality = quality_table(features)
    bounds = bounds_table(frame, task)
    outside = int(bounds.outside_range.sum())
    duplicate = int(features.duplicated().sum())
    outliers = outlier_table(features)
    n = Notebook(f'# Task {task}: {TITLES[task]}\n\nรายงาน EDA จากข้อมูลในเครื่อง วันที่ 2 พฤศจิกายน 2026 '
                 'ไม่มีการฝึกโมเดลใน notebook นี้ ค่าที่เขียนใน Markdown อิง snapshot ที่ตรวจครั้งนี้ '
                 'หากเปลี่ยนข้อมูลให้สร้างรายงานใหม่และรันทุก cell อีกครั้ง')
    n.md(f'## 1. Task Overview\n\n{OBJECTIVES[task]}\n\nTarget: `{target}`. '
         + ('0=ผู้เล่นปกติ, 1=ผู้เล่นที่โกง' if task == 1 else
            '0=Casual, 1=Grinder, 2=Social, 3=Whale' if task == 2 else
            'ยอดใช้จ่าย THB ใน 30 วันถัดไป' if task == 3 else '0=ปกติ, 1=น่าสงสัย แต่ไม่มี label ให้ตรวจ')
         + '\n\nไม่พบ metric ทางการในไฟล์ที่ได้รับ '
         + ('แนะนำ F1 และ ROC-AUC เป็น local proxy' if task == 1 else
            'แนะนำ macro F1 และ accuracy เป็น local proxy' if task == 2 else
            'แนะนำ RMSE และ MAE หน่วย THB เป็น local proxy' if task == 3 else
            'หากมี label ให้ตรวจ PR-AUC/F1 และ precision ตามงบตรวจสอบ ยังรายงานคะแนนเหล่านี้ไม่ได้'))
    n.code(SETUP + f'\ntrain, test, target, base = load_task(ROOT, {task})\n'
           'frame = train if train is not None else test\n'
           "features = frame.drop(columns=['player_id', target], errors='ignore')")
    n.md('## 2. Dataset Overview\n\n' + CARDS[task])
    n.code("""
    overview = pd.DataFrame([
        {'split': name, 'rows': len(data), 'columns': data.shape[1],
         'target_available': int(data[target].notna().sum()) if target in data else 0}
        for name, data in [('train', train), ('test', test)] if data is not None
    ])
    display(overview)
    display(features.dtypes.astype(str).value_counts().rename('feature_count').to_frame())
    with pd.option_context('display.max_rows', None):
        display(features.dtypes.astype(str).to_frame('dtype'))
    display(frame.drop(columns='player_id').head(3))
    """)
    label_count = int(train[target].notna().sum()) if train is not None else 0
    n.notes(f"{'Train มี ' + format(len(train), ',') + ' แถว; ' if train is not None else 'ไม่มีไฟล์ train; '}"
            f'test มี {len(test):,} แถว, feature {features.shape[1]} ตัว และ label ที่ใช้ได้ {label_count:,} แถว',
            'ต้องแยกความครบของ feature ออกจากความครบของ label',
            'ไม่เติม target ที่หายและไม่ใช้ player_id เป็น predictor',
            'ตรวจ missing และขอบเขตข้อมูลก่อนเลือก preprocessing')
    n.md('## 3. Data Quality Analysis')
    n.code(f"""
    quality = quality_table(features)
    display(quality.head(12))
    print('Duplicate IDs:', int(frame.player_id.duplicated().sum()))
    print('Duplicate feature rows:', int(features.duplicated().sum()))
    if train is not None:
        print('Train/test ID overlap:', len(set(train.player_id) & set(test.player_id)))
        print('Missing target:', int(train[target].isna().sum()))
    display(bounds_table(frame, {task}))
    display(outlier_table(features).head(8))
    quality.missing_pct.head(12).sort_values().plot.barh(xlabel='Missing (%)', title='Missing values')
    plt.tight_layout()
    plt.show()
    """)
    n.notes(f'Feature missing เฉลี่ย {missing:.2f}%; สูงสุด `{quality.index[0]}` '
            f'{quality.iloc[0].missing_pct:.2f}%; duplicate feature rows {duplicate:,}; '
            f'พบค่าที่อยู่นอกช่วงตรวจทั้งหมด {outside:,} จุด (นับระดับ cell); '
            f'feature ที่มี IQR outlier สูงสุดคือ `{outliers.index[0]}` {outliers.iloc[0].outlier_pct:.2f}%',
            'ค่าหลุด IQR เป็น outlier เชิงสถิติ ไม่ยืนยันว่าเป็นข้อมูลผิด '
            'ช่วงที่ตรวจมาจากความหมายใน data card และอาจสะท้อนนิยามหรือข้อมูลสังเคราะห์ที่ต่างออกไป',
            'ใช้โมเดลรองรับ missing หรือ fit imputer เฉพาะ training fold; ยังไม่ clip ทั้งชุด',
            'ตรวจ feature ที่อยู่นอกช่วงกับนิยามต้นทาง และเก็บ missing indicator เมื่อใช้ imputer')
    if train is not None:
        test_missing = test[features.columns].isna().mean() * 100
        missing_gap = (test_missing - quality.missing_pct).abs().sort_values(ascending=False)
        n.code("""
        missing_compare = pd.DataFrame({'train_pct': features.isna().mean() * 100,
                                        'test_pct': test[features.columns].isna().mean() * 100})
        missing_compare['gap_pp'] = missing_compare.test_pct - missing_compare.train_pct
        display(missing_compare.sort_values('gap_pp', key=lambda x: x.abs(), ascending=False).head(10))
        print('Duplicate test feature rows:', test[features.columns].duplicated().sum())
        train_hash = pd.util.hash_pandas_object(features, index=False)
        test_hash = pd.util.hash_pandas_object(test[features.columns], index=False)
        print('Train/test identical feature row hashes:', len(set(train_hash) & set(test_hash)))
        """)
        n.notes(f'Missing ใน test เฉลี่ย {test_missing.mean():.2f}%; '
                f'ช่องว่าง train/test ที่มากสุดคือ `{missing_gap.index[0]}` {missing_gap.iloc[0]:.2f} percentage points',
                'การเปรียบเทียบนี้ตรวจเฉพาะ missingness และ exact feature rows ยังไม่พิสูจน์ว่าไม่มี distribution shift',
                'ใช้ preprocessing ที่รองรับ missing ในทั้งสอง split และหลีกเลี่ยง fit บน test สำหรับ supervised tasks',
                'หากคะแนนจริงต่างจาก holdout มาก ให้ตรวจ feature distribution และรูปแบบเก็บข้อมูลเพิ่มเติม')
    n.md('## 4. Statistical Analysis')
    numeric = features.select_dtypes('number')
    plot_columns = numeric.columns[:4].tolist()
    n.code(f"""
    display(features.describe(include='all').T.head(12))
    features[{plot_columns!r}].hist(bins=35, figsize=(10, 6))
    plt.tight_layout()
    plt.show()
    """)
    descriptive_facts = '; '.join(f'`{column}` median={numeric[column].median():,.3f}, '
                                  f'p99={numeric[column].quantile(.99):,.3f}' for column in plot_columns)
    n.notes(f'ข้อมูลตัวเลขมี {numeric.shape[1]} feature; {descriptive_facts}',
            'ช่วงและรูปทรงของ feature ต่างกัน จึงต้องอ่านแกนและหน่วยก่อนเปรียบเทียบ',
            'Tree baseline ไม่ต้อง scale; หากเปลี่ยนเป็น linear/distance model ให้ scale ภายใน fold',
            'ตรวจ target และความสัมพันธ์กับ feature เพิ่มเพื่อเลือกการแปลงข้อมูล')
    if train is not None:
        n.code("""
        if target == 'spending_30d':
            display(frame[target].describe(percentiles=[.5, .9, .95, .99]).to_frame())
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            frame[target].hist(bins=50, ax=axes[0])
            axes[0].set_title('Spending (THB)')
            np.log1p(frame[target]).hist(bins=50, ax=axes[1])
            axes[1].set_title('log1p spending')
        else:
            counts = frame[target].value_counts(dropna=False).sort_index()
            display(pd.DataFrame({'count': counts, 'pct': counts / len(frame) * 100}))
            counts.plot.bar(title='Target distribution', ylabel='Rows')
        plt.tight_layout()
        plt.show()
        """)
        if task == 3:
            fact = (f'Target เป็นศูนย์ {(frame[target] == 0).sum():,} แถว '
                    f'({frame[target].eq(0).mean():.2%}), mean {frame[target].mean():,.2f}, '
                    f'median {frame[target].median():,.2f}, p99 {frame[target].quantile(.99):,.2f} THB')
            meaning = 'ยอดใช้จ่ายมีมวลที่ศูนย์และหางขวายาว ค่าเฉลี่ยห่างจาก median มาก'
            effect = 'RMSE จะได้รับผลจากผู้ใช้จ่ายสูงมาก; log target อาจช่วย metric อื่นแต่ไม่ควรเลือกก่อนรู้กติกา'
        else:
            counts = frame[target].value_counts()
            fact = f'จำนวน label แต่ละคลาสคือ {counts.sort_index().to_dict()}; target หาย {frame[target].isna().sum():,} แถว'
            meaning = 'คลาสมีจำนวนไม่เท่ากัน ต้องระวัง accuracy ที่กลบผลของคลาสเล็ก'
            effect = 'ใช้ stratified validation และตรวจ F1 รายคลาส; class weight ต้องประเมินจาก validation'
        n.notes(fact, meaning, effect, 'ใช้เฉพาะ label ที่สังเกตได้และรายงาน metric หลายมุม')
    n.md('## 5. Feature Relationships')
    pairs = strongest_pairs(features)
    pair_fact = f'คู่ feature ที่มี |Pearson r| สูงสุดคือ {pairs.index[0]} r={pairs.iloc[0]:.4f}'
    if train is not None:
        assoc = association(frame, target, regression=task == 3)
        top, value = assoc.index[0], assoc.iloc[0]
        n.code(f"""
        relevance = association(frame, target, regression={task == 3})
        display(relevance.head(12).rename('abs_spearman' if target == 'spending_30d' else 'eta_squared').to_frame())
        display(strongest_pairs(features).rename('pearson_r').to_frame())
        relevance.head(10).sort_values().plot.barh(title='Feature association with target')
        plt.tight_layout()
        plt.show()
        """)
        n.notes(f'Feature ที่มี association สูงสุดคือ `{top}` = {value:.4f} '
                + ('(|Spearman|)' if task == 3 else '(eta squared; ไม่สมมติลำดับคลาส)')
                + '; ' + pair_fact,
                'ความสัมพันธ์คำนวณบนคู่ที่ไม่ missing ไม่ได้แสดงเหตุและผลหรือยืนยัน leakage',
                'หลีกเลี่ยงใช้ correlation ของรหัส multiclass เป็นตัวชี้วัดหลัก; ยังไม่เลือก feature ด้วย validation ชุดสุดท้าย',
                'ตรวจ class-conditional patterns และเวลาที่ feature ถูกสร้างก่อน target')
    else:
        delta = period_changes(frame)
        top = delta.p99_abs_delta.idxmax()
        n.code("""
        display(strongest_pairs(features).rename('pearson_r').to_frame())
        deltas = period_changes(frame)
        display(deltas.sort_values('p99_abs_delta', ascending=False).head(12))
        columns = [f'login_count_{period}' for period in range(1, 5)]
        display(frame[columns].corr())
        """)
        n.notes(f'พบกลุ่ม feature วัดต่อเนื่อง {len(delta)} กลุ่ม; `{top}` มี p99 ของ |ช่วง4−ช่วง1| '
                f'{delta.loc[top, "p99_abs_delta"]:,.3f} ในหน่วยของตัวแปรนั้น; {pair_fact}',
                'feature หลายช่วงของบัญชีเดียวกันมีความสัมพันธ์กัน ขนาด delta ข้ามหน่วยเปรียบเทียบตรง ๆ ไม่ได้',
                'พิจารณา delta/std และ relative change โดยป้องกันหารศูนย์; ไม่มี label จึงหา target association ไม่ได้',
                'ตรวจแนวโน้มรายตัวแปรและเลือก feature summary โดยไม่เรียกแถวใดว่า anomaly จริง')
    n.md('### Feature interactions')
    if task == 1:
        interaction_code = """
        interaction_data = frame.assign(
            reaction_bin=pd.qcut(frame.reaction_time_ms, 4, duplicates='drop'),
            headshot_bin=pd.qcut(frame.headshot_percentage, 4, duplicates='drop'),
        )
        interaction = interaction_data.pivot_table(index='reaction_bin', columns='headshot_bin',
                                                  values=target, aggfunc='mean', observed=True)
        display(interaction)
        display(interaction_data.dropna(subset=[target]).groupby(
            ['reaction_bin', 'headshot_bin'], observed=True).size().unstack())
        """
        meaning = 'อัตรา cheater ในแต่ละช่องคำนวณจาก label ที่ทราบค่า ไม่ใช่ probability ที่โมเดลทำนาย'
    elif task == 2:
        interaction_code = """
        interaction_data = frame.assign(
            spending_bin=pd.qcut(frame.total_spending_thb, 4, duplicates='drop'),
            frequency_bin=pd.qcut(frame.play_frequency, 4, duplicates='drop'),
            whale=frame.segment.eq(3).astype(float),
        )
        interaction = interaction_data.pivot_table(index='spending_bin', columns='frequency_bin',
                                                  values='whale', aggfunc='mean', observed=True)
        display(interaction)
        display(pd.crosstab(interaction_data.spending_bin, interaction_data.frequency_bin))
        """
        meaning = 'ตารางแสดงสัดส่วน Whale ตาม spending และ play frequency ร่วมกัน ไม่สมมติว่า segment เป็นเลขต่อเนื่อง'
    elif task == 3:
        interaction_code = """
        interaction_data = frame.assign(
            previous_spending_bin=pd.qcut(frame.prev_month_spending, 4, duplicates='drop'),
        )
        interaction = interaction_data.pivot_table(index='segment', columns='previous_spending_bin',
                                                  values=target, aggfunc='median', observed=True)
        display(interaction)
        display(pd.crosstab(interaction_data.segment, interaction_data.previous_spending_bin))
        """
        meaning = 'ตารางแสดง median ยอดใช้จ่าย THB ตาม segment และยอดใช้จ่ายเดือนก่อนร่วมกัน'
    else:
        interaction_code = """
        login_delta = frame.login_count_4 - frame.login_count_1
        device_delta = frame.device_count_4 - frame.device_count_1
        interaction = pd.crosstab(pd.qcut(login_delta, 4, duplicates='drop'),
                                  pd.qcut(device_delta, 4, duplicates='drop'))
        display(interaction)
        """
        meaning = 'ตารางนับบัญชีที่มีการเปลี่ยน login และ device ร่วมกัน ไม่ใช่จำนวน anomaly ที่ยืนยันแล้ว'
    n.code(interaction_code)
    namespace = {'frame': frame, 'target': target, 'pd': pd, 'display': lambda *args: None}
    exec(textwrap.dedent(interaction_code), namespace)
    interaction = namespace['interaction']
    n.notes(f'ตาราง interaction มี {interaction.shape[0]}×{interaction.shape[1]} ช่อง; '
            f'ค่าที่สังเกตได้ต่ำสุด {interaction.min().min():,.4f}, สูงสุด {interaction.max().max():,.4f}',
            meaning + '; ช่วง bin มาจาก quartile ของข้อมูลที่ใช้วิเคราะห์ และบางช่องอาจมีตัวอย่างน้อย',
            'Tree model รองรับความสัมพันธ์ร่วมแบบนี้ได้; ยังไม่สร้างกฎตัดสินจาก EDA เพียงอย่างเดียว',
            'อ่านจำนวนตัวอย่างในแต่ละช่องก่อนตีความ และตรวจผลบน validation แยกจาก EDA')
    n.md('## 6. Task-Specific EDA')
    if task == 1:
        columns = ['reaction_time_ms', 'headshot_percentage', 'reports_received', 'kill_death_ratio']
        grouped = frame.groupby(target)[columns].median()
        n.code(f"""
        display(frame.groupby(target)[{columns!r}].median())
        missing_by_label = features.isna().mean(axis=1).groupby(frame[target]).mean()
        display(missing_by_label.rename('mean_missing_fraction').to_frame())
        display(frame.groupby(frame[target].isna())[{columns!r}].median())
        """)
        n.notes(f'Median reaction time: class 0={grouped.loc[0, "reaction_time_ms"]:.2f} ms, '
                f'class 1={grouped.loc[1, "reaction_time_ms"]:.2f} ms; '
                f'headshot: class 0={grouped.loc[0, "headshot_percentage"]:.2f}, '
                f'class 1={grouped.loc[1, "headshot_percentage"]:.2f}',
                'เป็นการเปรียบเทียบกลุ่มที่มี label; ไม่ควรสรุปว่าบัญชีที่ไม่มี label ปกติหรือโกง',
                'ฝึก supervised เฉพาะ known label และตรวจ false positive ที่พฤติกรรมคล้ายผู้เล่นเก่ง',
                'แยก unlabeled rows ไว้ ไม่ใช้ pseudo-label ใน baseline รอบแรก')
    elif task == 2:
        noise = assoc.reindex(['random_metric_1', 'random_metric_2', 'random_metric_3'])
        n.code("""
        display(frame.groupby(target)[['play_frequency', 'total_spending_thb', 'friend_count']].median())
        display(pd.crosstab(frame['player_type_tag'], frame[target], normalize='columns'))
        display(relevance.reindex(['random_metric_1', 'random_metric_2', 'random_metric_3']).to_frame())
        unseen = {column: sorted(set(test[column].dropna()) - set(train[column].dropna()))
                  for column in features.select_dtypes(include=['object', 'str'])}
        print('Unseen test categories:', unseen)
        """)
        n.notes(f'Noise feature มี eta squared {noise.round(6).to_dict()}; '
                f'ค่า association สูงสุดของ feature ทั้งหมด {value:.4f}',
                'data card ระบุ noise ไว้ชัดเจน; ความต่างของกลุ่มและ tag ยังไม่ยืนยันกระบวนการสร้าง target',
                'ตัด random_metric_1–3 จาก baseline; เข้ารหัส category โดยรองรับ unseen และ missing',
                'ตรวจ provenance ของ player_type_tag, engagement_level และ loyalty_tier ก่อนใช้งานจริง')
    elif task == 3:
        cap = int(frame[target].eq(frame[target].max()).sum())
        n.code("""
        spending_by_segment = frame.groupby('segment')[target].agg(['count', 'mean', 'median'])
        spending_by_segment['zero_fraction'] = frame.groupby('segment')[target].apply(lambda x: x.eq(0).mean())
        display(spending_by_segment)
        print('Rows at maximum target:', int(frame[target].eq(frame[target].max()).sum()))
        sample = frame[['prev_month_spending', target]].dropna().sample(3000, random_state=42)
        plt.scatter(np.log1p(sample.prev_month_spending), np.log1p(sample[target]), s=5, alpha=.2)
        plt.xlabel('log1p previous month spending')
        plt.ylabel('log1p next 30d spending')
        plt.tight_layout()
        plt.show()
        """)
        n.notes(f'ค่าสูงสุด {frame[target].max():,.2f} THB พบ {cap} แถว; '
                f'prev_month_spending กับ target มี Spearman {frame.prev_month_spending.corr(frame[target], method="spearman"):.4f}',
                'การซ้ำที่ค่าสูงสุดอาจเกิดจาก cap แต่เป็นสมมติฐาน ยังไม่มีหลักฐานยืนยัน '
                'historical spending ควรถูกบันทึกก่อนช่วง target',
                'ยังไม่ตัดยอดใช้จ่ายสูง; segment เป็น input ที่ให้มาและไม่ควรเชื่อม Task 2 ด้วยสมมติฐานว่าเวลาเดียวกัน',
                'ตรวจ timestamp, target cap และ metric ทางการ; ลอง two-stage หรือ log target ในรอบถัดไป')
    else:
        n.code("""
        display(frame[['password_changed', 'vpn_usage', 'suspicious_login_time', 'mass_item_sale']].mean().to_frame('observed_mean'))
        deltas = period_changes(frame)
        display(deltas.loc[['login_count', 'device_count', 'purchase_amount_mean', 'skill_rating']])
        delta_login = frame.login_count_4 - frame.login_count_1
        delta_login.hist(bins=50)
        plt.xlabel('Login count: period 4 - period 1')
        plt.ylabel('Accounts')
        plt.tight_layout()
        plt.show()
        """)
        observed_delta = frame.login_count_4 - frame.login_count_1
        n.notes(f'Login delta median={observed_delta.median():.3f}, '
                f'p99 |delta|={observed_delta.abs().quantile(.99):.3f}; '
                f'vpn_usage mean={frame.vpn_usage.mean():.4f} คำนวณจากค่าที่ไม่ missing',
                'flag ที่ให้มาเป็น input ไม่ใช่ ground truth ของ is_anomaly และค่าเฉลี่ยของ flag ไม่ใช่ anomaly rate',
                'ใช้ทั้งระดับพฤติกรรมและความเปลี่ยนแปลง; ห้ามรายงาน F1 หรือ accuracy เมื่อไม่มี label',
                'เริ่ม IsolationForest แล้วทดสอบความไวของ threshold ภายหลัง; ต้องขอ label เพื่อยืนยันคุณภาพ')
    n.md('## 7. EDA Findings & Insights')
    n.md(f'- Feature missing เฉลี่ย {missing:.2f}% และมีค่าที่อยู่นอกช่วงตรวจ {outside:,} จุด\n'
         f'- Duplicate feature rows {duplicate:,} แถว\n'
         + (f'- Train มี target ที่ทราบค่า {label_count:,} จาก {len(train):,} แถว\n' if train is not None else
            '- มีเฉพาะ test ไม่มี label จึงยังสรุปความแม่นยำของ anomaly detection ไม่ได้\n')
         + '- ความสัมพันธ์เป็น observed association; leakage และสาเหตุของค่าผิดช่วงยังเป็นประเด็นที่ต้องตรวจ')
    n.md('## 8. Preprocessing Recommendations')
    n.md('- ไม่ใช้ player_id เป็น predictor และไม่เติม target ที่หาย\n'
         '- Fit imputer/encoder เฉพาะ training fold; HistGradientBoosting รองรับ NaN ใน numerical features\n'
         '- ยังไม่ลบ IQR outlier อัตโนมัติ เพราะอาจเป็นกลุ่มพฤติกรรมที่ต้องการทำนาย\n'
         + ('- ใช้ categorical-aware encoding และตัด noise feature ที่ระบุใน data card\n' if task == 2 else '')
         + ('- ตรวจ log1p ของ spending และเพิ่ม zero indicator เป็นตัวเลือกหลังยืนยัน metric\n' if task == 3 else '')
         + ('- เพิ่ม delta/std ของ 4 ช่วงเวลา; median imputation พร้อม indicator; แยก NaN จากศูนย์\n' if task == 5 else '')
         + '- Tree model ไม่จำเป็นต้อง scale; ถ้าใช้ linear/distance model ให้ scale ภายใน fold')
    n.md('## 9. Baseline Model Recommendations')
    n.md(('เริ่ม HistGradientBoostingClassifier กับ known labels แบ่ง stratified holdout 80/20 seed 42 '
          'ตรวจ F1/ROC-AUC และ confusion matrix' if task == 1 else
          'เริ่ม HistGradientBoostingClassifier พร้อม categorical features แบ่ง stratified holdout 80/20 seed 42 '
          'ตรวจ macro F1 และ accuracy รายคลาส' if task == 2 else
          'เริ่ม HistGradientBoostingRegressor บน THB โดยตรง แบ่ง holdout 80/20 seed 42 '
          'เทียบ RMSE และ MAE กับ mean predictor; หากมี timestamp ให้เปลี่ยนเป็น temporal holdout' if task == 3 else
          'เริ่ม IsolationForest บนข้อมูล unlabeled พร้อม delta/std; ไม่มี supervised validation '
          'การ fit บน test เป็น transductive ต้องตรวจข้อกำหนดการแข่งขันก่อนส่ง; top 10% เป็นสมมติฐานเริ่มต้นเท่านั้น')
         + '\n\nการฝึกโมเดลอยู่ใน `02_baseline.ipynb` ของ task นี้ ไม่มีการ fit ใน EDA notebook นี้')
    n.md('## 10. Final EDA Summary')
    n.md(f'ยืนยันจากข้อมูลในเครื่อง: {len(frame):,} แถวที่ใช้วิเคราะห์, {features.shape[1]} feature, '
         f'missing เฉลี่ย {missing:.2f}% และ known labels {label_count:,} แถว\n\n'
         'ขั้นถัดไปคือรัน baseline แยกจาก EDA ตรวจ prediction และเก็บ local validation '
         'ก่อนปรับโมเดลให้ยืนยัน metric ทางการ วิธีจับคู่แถว submission และ provenance ของ feature '
         'ยังไม่สามารถพิสูจน์ temporal leakage หรือสาเหตุที่ข้อมูลบางค่าหลุดช่วงจากไฟล์ปัจจุบันได้')
    n.save(ROOT / TASKS[task][2] / '01_eda.ipynb')


def build_images():
    train, test, target, base = load_task(ROOT, 4)
    inv = pd.read_csv(ROOT / 'artifacts/image_train_inventory.csv')
    test_inv = pd.read_csv(ROOT / 'artifacts/image_test_inventory.csv')
    available = int(inv.readable.sum())
    joined = train.merge(inv, on='id', validate='one_to_one')
    counts = joined.groupby('class').readable.agg(['sum', 'count'])
    dup = int(inv.loc[inv.readable, 'sha256'].duplicated().sum())
    overlap = len(set(inv.loc[inv.readable, 'sha256']) & set(test_inv.loc[test_inv.readable, 'sha256']))
    n = Notebook('# Task 4: Game title detection\n\nรายงาน EDA จากภาพในเครื่อง วันที่ 2 พฤศจิกายน 2026 ไม่มีการฝึกโมเดล')
    n.md('## 1. Task Overview\n\n' + OBJECTIVES[4] + '\n\nTarget คือ `class` จำนวน 5 เกม '
         'ไม่พบ metric ทางการ แนะนำ macro F1 และ accuracy เป็น local proxy')
    n.code(SETUP + '\ntrain, test, target, base = load_task(ROOT, 4)')
    n.md('## 2. Dataset Overview\n\n' + CARDS[4])
    n.code("""
    display(pd.DataFrame({'split': ['train', 'test'], 'rows': [len(train), len(test)],
                          'labels': [train['class'].notna().sum(), test['class'].notna().sum()]}))
    display(train.drop(columns='id').head())
    display(train.dtypes.astype(str).to_frame('dtype'))
    """)
    n.notes(f'Train CSV มี {len(train):,} แถว และ test {len(test):,} แถว; test class ไม่มีค่าทุกแถว',
            'จำนวนแถวใน manifest ไม่รับประกันว่ามีไฟล์ภาพครบ', 'ตรวจ path จริงก่อนสร้าง training set',
            'ตรวจการมีอยู่ ความอ่านได้ ขนาด และ mode ของไฟล์ทั้งหมด')
    n.md('## 3. Data Quality Analysis')
    n.code("""
    train_inventory = image_inventory(base, train, 'train')
    test_inventory = image_inventory(base, test, 'test')
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    train_inventory.to_csv(artifacts / 'image_train_inventory.csv', index=False)
    test_inventory.to_csv(artifacts / 'image_test_inventory.csv', index=False)
    quality = pd.DataFrame([
        {'split': split, 'manifest_rows': len(inventory), 'exists': inventory.exists.sum(),
         'readable': inventory.readable.sum(),
         'corrupt': (inventory.exists & ~inventory.readable).sum()}
        for split, inventory in [('train', train_inventory), ('test', test_inventory)]
    ])
    display(quality)
    print('Duplicate train IDs:', train.id.duplicated().sum())
    print('Duplicate test IDs:', test.id.duplicated().sum())
    """)
    n.notes(f'Train อ่านได้ {available:,}/{len(train):,} ภาพ ({available / len(train):.2%}); '
            f'ขาด {int((~inv.exists).sum()):,} ภาพ; test อ่านได้ {test_inv.readable.sum():,}/{len(test):,}; '
            f'ไฟล์ที่มีอยู่แต่อ่านไม่ได้ {int((inv.exists & ~inv.readable).sum() + (test_inv.exists & ~test_inv.readable).sum()):,}',
            'ข้อมูลฝึกภาพมีข้อจำกัดรุนแรง แต่ข้อมูล test ต้องทำนายครบทุกแถว',
            'ใช้เฉพาะ readable train; ห้ามเติมภาพว่างแทนภาพฝึกที่ขาดเพราะจะสร้างสัญญาณเทียม',
            'กู้ไฟล์ train ที่ขาดจากแหล่งข้อมูลเดิมในรอบถัดไป')
    n.md('## 4. Statistical Analysis')
    n.code("""
    availability = train.merge(train_inventory, on='id', validate='one_to_one')
    counts = availability.groupby('class').readable.agg(['sum', 'count'])
    counts.columns = ['available', 'manifest']
    counts['available_pct'] = counts.available / counts.manifest * 100
    display(counts)
    counts[['manifest', 'available']].plot.bar(title='Images per class', ylabel='Images')
    plt.tight_layout()
    plt.show()
    display(pd.concat([train_inventory.assign(split='train'), test_inventory.assign(split='test')])
            .query('readable').groupby(['split', 'width', 'height', 'mode']).size().to_frame('count'))
    """)
    n.notes(f'ภาพ train ที่อ่านได้ต่อคลาสคือ {counts["sum"].to_dict()}; '
            f'อัตราภาพที่มีต่อคลาสอยู่ที่ {(counts["sum"] / counts["count"]).min():.2%} ถึง '
            f'{(counts["sum"] / counts["count"]).max():.2%}',
            'แต่ละคลาสมีภาพขาดมากและจำนวนตัวอย่างไม่เท่ากัน แม้อัตราภาพที่เหลือใกล้เคียงกัน '
            'ก็ยังยืนยันไม่ได้ว่า missing เป็นแบบสุ่ม',
            'ใช้ stratified split และ macro F1; รายงานคะแนนบน available subset อย่างชัดเจน',
            'ตรวจภาพตัวอย่างและความซ้ำก่อนแบ่ง validation')
    n.md('## 5. Feature Relationships')
    n.code("""
    train_hashes = train_inventory.loc[train_inventory.readable, 'sha256']
    test_hashes = test_inventory.loc[test_inventory.readable, 'sha256']
    print('Repeated train file hashes:', train_hashes.duplicated().sum())
    print('Train/test exact file hash intersection:', len(set(train_hashes) & set(test_hashes)))
    print('Train/test ID intersection:', len(set(train.id) & set(test.id)))
    readable = availability.query('readable')
    conflicts = readable.groupby('sha256')['class'].nunique()
    print('Hashes assigned multiple labels:', conflicts.gt(1).sum())
    """)
    n.notes(f'พบ repeated SHA256 ใน train {dup:,} แถว และ hash ตรงกันระหว่าง train/test {overlap:,} ค่า',
            'การตรวจนี้ครอบคลุม exact file duplicates แต่ไม่จับภาพที่ถูก re-encode/crop หรือเฟรมใกล้กัน',
            'random validation อาจยังมี near duplicate leakage ถึงแม้ exact hash จะไม่ตรงกัน',
            'ตรวจ perceptual hash หรือ group ตามวิดีโอต้นทางในรอบถัดไป')
    n.md('## 6. Task-Specific EDA')
    n.code("""
    fig, axes = plt.subplots(5, 3, figsize=(10, 10))
    for row, label in enumerate(sorted(readable['class'].unique())):
        examples = readable.loc[readable['class'].eq(label)].sample(3, random_state=42)
        for axis, image_id in zip(axes[row], examples.id):
            with Image.open(base / 'train' / f'{image_id}.jpg') as image:
                axis.imshow(image)
            axis.set_title(label, fontsize=9)
            axis.axis('off')
    plt.tight_layout()
    plt.show()
    colors = []
    for image_id, label in zip(readable.id, readable['class']):
        with Image.open(base / 'train' / f'{image_id}.jpg') as image:
            rgb = np.asarray(image.convert('RGB').resize((24, 12)), dtype=float)
        colors.append({'class': label, 'red': rgb[:, :, 0].mean(),
                       'green': rgb[:, :, 1].mean(), 'blue': rgb[:, :, 2].mean()})
    color_summary = pd.DataFrame(colors).groupby('class').mean()
    display(color_summary)
    """)
    # Calculate the same color statistics to write observed, rather than guessed, interpretation.
    from PIL import Image
    colors = []
    for row in joined.loc[joined.readable].itertuples(index=False):
        with Image.open(base / 'train' / f'{row.id}.jpg') as image:
            rgb = np.asarray(image.convert('RGB').resize((24, 12)), dtype=float)
        colors.append((row[1], *rgb.mean(axis=(0, 1))))
    color_summary = pd.DataFrame(colors, columns=['class', 'red', 'green', 'blue']).groupby('class').mean()
    spread = color_summary.max() - color_summary.min()
    n.notes(f'ค่า mean RGB ระดับคลาสต่างกันสูงสุดระหว่างคลาส: R={spread.red:.2f}, '
            f'G={spread.green:.2f}, B={spread.blue:.2f} บนสเกล 0–255; ภาพตัวอย่างสุ่มคลาสละ 3 ภาพ '
            'พบภาพเอียง สี/ความสว่างต่างกันมาก และ noise ที่มองเห็นได้ในหลายภาพตัวอย่าง',
            'สีและองค์ประกอบภาพอาจให้สัญญาณเบื้องต้น แต่ค่าเฉลี่ยสีไม่ได้ยืนยันว่าจำแนกเกมได้ดี '
            'ยังไม่ทราบว่าลักษณะภาพเกิดจาก augmentation หรือภาพต้นฉบับ',
            'ใช้ resized pixels/color histogram เป็น baseline ราคาถูก และตัดสินจาก validation '
            'ความไวต่อการหมุนและเปลี่ยนสีเป็นข้อจำกัดที่ควรตรวจต่อ',
            'เก็บภาพต้นฉบับไว้สำหรับ transfer learning เมื่อข้อมูลครบ')
    n.md('## 7. EDA Findings & Insights\n\n'
         f'- ภาพ train ใช้ได้ {available:,} ภาพ จาก manifest {len(train):,} แถว\n'
         f'- Test อ่านได้ {test_inv.readable.sum():,} ภาพ\n'
         f'- Exact train/test file hash overlap {overlap:,} ค่า แต่ยังไม่ตรวจ near duplicates\n'
         '- ความขาดของ train เป็นข้อจำกัดหลักของคะแนนและความเป็นตัวแทนของ validation')
    n.md('## 8. Preprocessing Recommendations\n\n'
         '- กรอง readable train ก่อน split และคงลำดับ test ตาม CSV\n'
         '- แปลง RGB, resize คงที่ และหาร pixel ด้วย 255; color histogram ใช้สัดส่วนในแต่ละ bin\n'
         '- ใช้ mapping ตามตาราง data card ไม่ใช้ alphabetical label encoding\n'
         '- ยังไม่ใช้ augmentation เพื่อให้ baseline ตรวจสอบง่าย; ไม่แทนภาพหายด้วยภาพว่าง')
    n.md('## 9. Baseline Model Recommendations\n\n'
         'เทียบ ExtraTrees กับ SVM หลัง StandardScaler จาก pixel 24×12 และ RGB histogram บน CPU '
         'แบ่ง readable subset แบบ stratified 80/20 seed 42 ตรวจ macro F1, accuracy และ confusion matrix '
         'ความท้าทายคือข้อมูลฝึกขาดและ screenshot ใกล้กันอาจทำให้ validation สูงเกินจริง '
         'การฝึกอยู่ใน `02_baseline.ipynb` ของ task นี้')
    n.md('## 10. Final EDA Summary\n\n'
         f'ยืนยันว่าภาพฝึกอ่านได้ {available:,} ภาพ โดยมีครบ 5 คลาส แต่ขาด {len(train) - available:,} ภาพ '
         'ขั้นถัดไปคือสร้าง baseline จาก subset นี้และกู้ภาพที่ขาด '
         'ยังไม่ทราบเหตุที่ไฟล์ขาด แหล่งวิดีโอของภาพ และ metric ทางการ '
         'ไม่ควรตีความคะแนนบน subset ว่าแทนข้อมูลทั้งหมดได้')
    n.save(ROOT / 'task4_images/01_eda.ipynb')


def build_audit():
    rows = []
    for task in TASKS:
        train, test, target, base = load_task(ROOT, task)
        rows.append({'task': task, 'train_rows': len(train) if train is not None else 0,
                     'test_rows': len(test), 'known_labels': int(train[target].notna().sum()) if train is not None else 0})
    audit = pd.DataFrame(rows)
    inv = pd.read_csv(ROOT / 'artifacts/image_train_inventory.csv')
    n = Notebook('# Data audit และแผน 1 ชั่วโมง\n\n'
                 'เป้าหมาย: EDA ที่อ่านได้ใน notebook และ baseline submission ครบ 5 งาน '
                 'ใช้ข้อมูลในเครื่อง ณ 2 พฤศจิกายน 2026 โค้ดใช้ชื่ออังกฤษ คำอธิบายภาษาไทย')
    n.md('## แผนการทำงาน\n\n'
         '| เวลา | งาน | ผลลัพธ์ |\n|---|---|---|\n'
         '| 0–10 นาที | audit schema, label, image files และ sample submission | ข้อจำกัดและข้อมูลพร้อมใช้ |\n'
         '| 10–25 นาที | EDA คุณภาพข้อมูล distribution และความสัมพันธ์ | 01_eda.ipynb ทั้ง 5 งาน |\n'
         '| 25–45 นาที | train/validation และ refit baseline บน CPU | prediction ครบทุกงาน |\n'
         '| 45–60 นาที | ตรวจไฟล์ อ่าน notebook outputs และสรุปข้อจำกัด | baseline_submission.csv |\n\n'
         'ไม่ทำ hyperparameter search ในรอบแรก; การฝึกแยกอยู่ใน 02_baseline.ipynb ของแต่ละ task '
         'ข้อค้นพบทั้งหมดอยู่ใน notebook ไม่สร้างรายงาน findings แยก')
    n.code(SETUP)
    n.md('## ตรวจไฟล์และ label ทุกงาน')
    n.code("""
    rows = []
    test_ids = {}
    for task in TASKS:
        train, test, target, base = load_task(ROOT, task)
        identifier = 'id' if task == 4 else 'player_id'
        test_ids[task] = test[identifier]
        rows.append({
            'task': task, 'train_rows': len(train) if train is not None else 0,
            'test_rows': len(test), 'test_columns': test.shape[1],
            'known_labels': int(train[target].notna().sum()) if train is not None else 0,
            'missing_train_labels': int(train[target].isna().sum()) if train is not None else 0,
            'duplicate_test_ids': int(test[identifier].duplicated().sum()),
            'train_test_id_overlap': len(set(train[identifier]) & set(test[identifier])) if train is not None else 0,
        })
    audit = pd.DataFrame(rows)
    display(audit)
    """)
    n.notes(f'Test ทุกงานมี 25,889 แถว; known labels ของ Task 1–5 คือ {audit.known_labels.tolist()}',
            'Task 1 มี label หาย ส่วน Task 5 ไม่มี train; ขนาด test เท่ากันยังไม่แปลว่าใช้ player ID ลำดับเดียวกัน',
            'แยก supervised training ตาม target ที่มีจริง และใช้ unsupervised baseline ใน Task 5',
            'ตรวจ sample schema และลำดับแถวก่อนรวมคำตอบ')
    n.md('## ตรวจ sample submission และ row mapping')
    n.code("""
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    display(sample.head(6))
    display(sample.notna().sum().rename('non_missing').to_frame())
    print('Task 1/2 same player_id at same row:', int(test_ids[1].eq(test_ids[2]).sum()))
    print('Task 1/3 same player_id at same row:', int(test_ids[1].eq(test_ids[3]).sum()))
    assert sample.columns.tolist() == ['id', 'task1', 'task2', 'task3', 'task4', 'task5']
    assert sample.id.is_unique and len(sample) == 25889
    """)
    n.notes('Sample ใช้ ANS ID และมีคำตอบตัวอย่างเพียง 4 แถวต่อ task; '
            'ไม่มีคอลัมน์เชื่อม ANS ID กับ player_id หรือ image id',
            'ค่าตัวอย่างไม่ใช่ training label และลำดับ test ของแต่ละงานไม่ควรถูก sort หรือ merge เอง',
            'Baseline ใช้แถวที่ i ของ test งานนั้นเป็นแถวที่ i ของ submission ตามสมมติฐาน positional mapping',
            'รักษา sample.id เดิมและยืนยันกติกา positional mapping กับผู้จัดก่อนอัปโหลด')
    n.md('## ตรวจข้อมูลภาพ')
    n.code("""
    image_base = ROOT / TASKS[4][0]
    image_train = pd.read_csv(image_base / 'train.csv')
    image_test = pd.read_csv(image_base / 'test.csv')
    rows = []
    for split, data in [('train', image_train), ('test', image_test)]:
        exists = data.id.map(lambda image_id: (image_base / split / f'{image_id}.jpg').exists())
        rows.append({'split': split, 'manifest_rows': len(data), 'existing_images': exists.sum(),
                     'missing_images': (~exists).sum()})
    display(pd.DataFrame(rows))
    """)
    n.notes(f'ภาพฝึกมีไฟล์ {inv.exists.sum():,} ภาพ จาก 31,546 รายการ; ภาพ test มีครบ 25,889 ภาพ',
            'จำนวน label ใน CSV สูงกว่าจำนวนภาพที่ฝึกได้จริงมาก',
            'กรองภาพอ่านได้ก่อน split โดยตรวจ full decode ใน Task 4 EDA',
            'สร้าง image baseline จาก subset พร้อมระบุ selection bias และกู้ไฟล์ในรอบถัดไป')
    n.md('## สรุปข้ามงานและขั้นถัดไป\n\n'
         '| งาน | ประเด็นที่ยืนยัน | แนวทาง baseline |\n|---|---|---|\n'
         '| 1 Anti-cheat | known labels 90,816; หาย 13,184 | HistGradientBoosting, stratified holdout |\n'
         '| 2 Segmentation | 4 คลาส มี numerical/categorical และ noise ตาม data card | categorical encoding + HistGradientBoosting |\n'
         '| 3 Spending | มีค่า 0 จำนวนมากและหางขวายาว | HistGradientBoostingRegressor, เทียบ mean baseline |\n'
         f'| 4 Images | ภาพ train อ่านได้ {inv.readable.sum():,}/31,546 | pixel/histogram + ExtraTrees หรือ SVM |\n'
         '| 5 Anomaly | ไม่มี train/label มี 4 ช่วงเวลา | delta/std + IsolationForest |\n\n'
         'ข้อจำกัดร่วม: ยังไม่มี metric ทางการ, row mapping ที่ระบุโดยผู้จัด, timestamp และ provenance '
         'Task 5 ใช้ test เพื่อ fit แบบ transductive และ top 10% เป็นสมมติฐาน ต้องยืนยันกติกาก่อนส่ง '
         'ทุก EDA notebook มี findings อยู่ใต้การวิเคราะห์ และเก็บ output เมื่อรันสำเร็จ '
         'ไฟล์ submission อยู่ใน submissions/baseline_submission.csv หลังรัน baseline สำเร็จ '
         'ไม่มีการอัปโหลด Kaggle อัตโนมัติ')
    n.save(ROOT / '00_data_audit.ipynb')


if __name__ == '__main__':
    build_audit()
    for task in [1, 2, 3, 5]:
        build_tabular(task)
    build_images()
