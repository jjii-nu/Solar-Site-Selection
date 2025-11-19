import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Patch
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows/Mac/Colab 환경에 맞춰 설정 필요)
# Colab의 경우 별도 폰트 설치 필요. 로컬 환경이라 가정하고 NanumGothic 설정
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 로드 (데이터 파일이 같은 경로에 있다고 가정)
try:
    df = pd.read_csv('climate_data.csv')
except FileNotFoundError:
    # 테스트를 위한 더미 데이터 생성 (파일이 없을 경우 실행됨)
    print("csv 파일을 찾을 수 없어 더미 데이터를 생성합니다.")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    locs = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주',
            '양구', '포천', '속초', '옹진', '가평', '홍천', '강릉', '동해', '수원', '이천', '원주', '정선', '태백',
            '아산', '천안', '청주', '충주', '영주', '영양', '보령', '공주', '옥천', '상주', '안동', '청송', '포항',
            '익산', '무주', '성주', '부안', '정읍', '합천', '밀양', '울주', '나주', '화순', '순천', '사천', '창원', '해남', '장흥', '여수']
    data = []
    for m in range(1, 13):
        for loc in locs:
            # 여름에 높고 겨울에 낮게 랜덤 데이터 생성
            base = 10 if m in [12, 1, 2] else (20 if m in [6, 7, 8] else 15)
            val = base + np.random.rand() * 10
            data.append({'월': m, '지점명': loc, '일사량': val, '시군구': loc + '시'}) # 시군구 더미
    df = pd.DataFrame(data)


# 격자 좌표 정의 (7행 x 7열의 불규칙 격자)
grid_mapping = [
    # Row 0
    [(None, None), (None, None), (None, None), ('양구', '양구군'), ('양구', '양구군'),  (None, None), (None, None)],
    # Row 1
    [(None, None), ('포천', '포천시'), ('포천', '포천시'), ('양구', '양구군'), ('고성', '고성군'), (None, None), (None, None)],
    # Row 2
    [('옹진', '옹진군'), ('양주', '양주시'), ('가평', '가평군'), ('홍천', '홍천군'), ('강릉', '강릉시'), ('동해', '동해시'), (None, None)],
    # Row 3
    [('옹진', '옹진군'), ('수원', '수원시'), ('이천', '이천시'), ('원주', '원주시'), ('정선', '정선군'), ('태백', '태백시'), (None, None)],
    # Row 4
    [('아산', '아산시'), ('천안', '천안시'), ('청주', '청주시'), ('충주', '충주시'), ('영주', '영주시'), ('영양', '영양군'),(None, None)],
    # Row 5
    [('보령', '보령시'), ('공주', '공주시'), ('옥천', '옥천군'), ('상주', '상주시'), ('안동', '안동시'), ('청송', '청송군'), ('포항', '포항시')],
    # Row 6
    [('익산', '익산시'), ('익산', '익산시'), ('무주', '무주군'), ('성주', '성주군'), ('대구', '달성군'), ('포항', '포항시'), ('포항', '포항시')],
    # Row 7
    [('부안', '부안군'), ('정읍', '정읍시'), ('무주', '무주군'), ('합천', '합천군'), ('밀양', '밀양시'), ('울주', '울주군'), ('울주', '울주군')],
    # Row 8
    [('나주', '나주시'), ('화순', '화순군'), ('순천', '순천시'), ('사천', '사천시'), ('창원', '창원시'), ('창원', '창원시'),(None, None)],
    # Row 9
    [('해남', '해남군'), ('장흥', '장흥군'), ('여수', '여수시'), ('사천', '사천시'), (None, None),(None, None), (None, None)]
]

# 격자 셀 크기 및 시작 위치
cell_size = 50000  # 50km
start_x = 900000
start_y = 1600000

# 격자 중심점 좌표 계산 및 매핑
grid_points = []
for row_idx, row in enumerate(grid_mapping):
    for col_idx, cell in enumerate(row):
        if cell[0] is not None:
            center_x = start_x + col_idx * cell_size + cell_size / 2
            center_y = start_y + (len(grid_mapping) - 1 - row_idx) * cell_size + cell_size / 2

            grid_points.append({
                'row': row_idx,
                'col': col_idx,
                'center_x': center_x,
                'center_y': center_y,
                'name': cell[0],
                'sigungu': cell[1]
            })

grid_df = pd.DataFrame(grid_points)

# 데이터 매칭 함수
def match_location(row):
    sigungu = row['시군구'].strip() if isinstance(row['시군구'], str) else ''
    if sigungu == '포항시': return '포항'
    elif sigungu == '달성군': return '대구'
    elif sigungu == '기장군': return '부산'
    else:
        return sigungu.replace('시', '').replace('군', '')

df['matched_name'] = df.apply(match_location, axis=1)


# ============================================================================
# [수정 1] 1년치 데이터를 통합하여 Global K-Means 군집화 수행
# ============================================================================
print("전역(Global) 군집화 수행 중...")

# 1. 격자 지도에 표시될 수 있는 모든 데이터 수집 (월별, 지역별 전부)
all_valid_data = []
valid_locations = set(grid_df['name'].unique())

# 데이터프레임에서 격자에 매칭되는 데이터만 추출
filtered_df = df[df['matched_name'].isin(valid_locations)].copy()
if filtered_df.empty:
    raise ValueError("매칭되는 데이터가 없습니다. CSV 파일의 시군구 명칭을 확인해주세요.")

X_global = filtered_df[['일사량']].values

# 2. K-Means 수행 (전체 데이터 기준)
n_clusters = 12
kmeans_global = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_global.fit(X_global)

# 3. 군집 순서 정렬 (일사량이 적은 순 -> 많은 순)
# cluster_centers_는 [[val1], [val2], ...] 형태이므로 flatten()
sorted_idx = np.argsort(kmeans_global.cluster_centers_.flatten())
# 예: sorted_idx가 [2, 0, 4, 1, 3]이면 2번 군집이 가장 값이 작음 -> 이를 0번(보라색)으로 매핑

# 기존 라벨을 정렬된 라벨(0~4)로 변환하는 맵 생성
label_map = {old_label: new_label for new_label, old_label in enumerate(sorted_idx)}

# 4. 범례(Legend) 생성을 위한 각 군집의 범위 계산
cluster_ranges = {}
filtered_df['global_cluster'] = kmeans_global.predict(X_global)
filtered_df['sorted_cluster'] = filtered_df['global_cluster'].map(label_map)

for i in range(n_clusters):
    cluster_vals = filtered_df[filtered_df['sorted_cluster'] == i]['일사량']
    if not cluster_vals.empty:
        cluster_ranges[i] = (cluster_vals.min(), cluster_vals.max(), cluster_vals.mean())
    else:
        cluster_ranges[i] = (0, 0, 0)

# ============================================================================
# [수정 2] 색상 설정 (보라색 -> 노란색)
# ============================================================================
# matplotlib의 'viridis' 컬러맵 사용 (0에 가까울수록 보라, 1에 가까울수록 노랑)
cmap = plt.cm.viridis
# 0~4 레벨에 맞는 색상 추출
cluster_colors = [cmap(i / (n_clusters - 1)) for i in range(n_clusters)]

# ============================================================================
# 시각화
# ============================================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 16)) # 범례 공간 확보를 위해 높이 조절
fig.suptitle('월별 일사량 (연간 통합 군집화 적용)', fontsize=20, fontweight='bold')

for month in range(1, 13):
    ax = axes[(month-1)//4, (month-1)%4]
    
    # 해당 월 데이터 추출
    month_data = df[df['월'] == month].copy()
    
    # 대한민국 외곽선 (간단한 박스)
    korea_outline_x = [start_x, start_x + 7*cell_size, start_x + 7*cell_size, start_x, start_x]
    korea_outline_y = [start_y, start_y, start_y + 10*cell_size, start_y + 10*cell_size, start_y]
    ax.plot(korea_outline_x, korea_outline_y, 'k-', linewidth=1.5, alpha=0.3)
    
    # 격자 그리기
    for row_idx in range(len(grid_mapping) + 1):
        y = start_y + row_idx * cell_size
        ax.plot([start_x, start_x + 7*cell_size], [y, y], color='gray', linewidth=0.5, alpha=0.3)
    for col_idx in range(8):
        x = start_x + col_idx * cell_size
        ax.plot([x, x], [start_y, start_y + 10*cell_size], color='gray', linewidth=0.5, alpha=0.3)

    # 데이터 매핑 및 그리기
    for _, grid_point in grid_df.iterrows():
        matched = month_data[month_data['matched_name'] == grid_point['name']]
        
        if len(matched) > 0:
            solar = matched.iloc[0]['일사량']
            
            # [중요] 전역 모델로 예측하여 색상 결정
            original_label = kmeans_global.predict([[solar]])[0]
            sorted_label = label_map[original_label]
            
            # 사각형 그리기
            rect_x = start_x + grid_point['col'] * cell_size
            rect_y = start_y + (len(grid_mapping) - 1 - grid_point['row']) * cell_size
            
            rect = Rectangle((rect_x, rect_y), cell_size, cell_size,
                             facecolor=cluster_colors[sorted_label], 
                             alpha=0.8, edgecolor='none')
            ax.add_patch(rect)
            
            # [수정 3] 파란색 중심점('bo') 제거함
            # 텍스트로 값을 표시하고 싶다면 아래 주석 해제
            # ax.text(grid_point['center_x'], grid_point['center_y'], f"{solar:.1f}", 
            #         ha='center', va='center', fontsize=7, color='white')

    ax.set_xlim(start_x - 10000, start_x + 7*cell_size + 10000)
    ax.set_ylim(start_y - 10000, start_y + 10*cell_size + 10000)
    ax.set_aspect('equal')
    ax.set_title(f'{month}월', fontsize=14, fontweight='bold')
    ax.axis('off') # 축 눈금 제거하여 깔끔하게

# ============================================================================
# [수정 2-2] 하단 범례(Legend) 추가
# ============================================================================
legend_elements = []
for i in range(n_clusters):
    min_val, max_val, mean_val = cluster_ranges[i]
    label_text = f'Level {i+1}: {min_val:.1f} ~ {max_val:.1f} (평균 {mean_val:.1f})'
    legend_elements.append(Patch(facecolor=cluster_colors[i], edgecolor='gray', label=label_text))

# 범례를 그림 하단 중앙에 배치
fig.legend(handles=legend_elements, loc='lower center', 
           bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=12, title="일사량 군집 범위 (Low -> High)")

plt.tight_layout()
# 범례 공간 확보를 위해 하단 여백 조정
plt.subplots_adjust(bottom=0.1) 

plt.savefig('solar_radiation_global_kmeans.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 연간 통합 군집 통계 ===")
for i in range(n_clusters):
    min_v, max_v, mean_v = cluster_ranges[i]
    print(f"군집 {i+1} (색상지수 {i}): 평균={mean_v:.2f}, 범위=[{min_v:.2f}, {max_v:.2f}]")