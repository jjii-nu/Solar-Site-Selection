import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드 및 전처리 (최고기온 데이터 준비)
try:
    df = pd.read_csv('climate_data.csv')
    # 만약 csv에 최고기온 컬럼이 없다면 에러가 날 수 있으므로 확인 필요
    if '최고기온' not in df.columns and '일사량' in df.columns:
         # 파일은 있는데 컬럼이 없는 경우 (기존 파일 재사용 시)
         raise ValueError("CSV에 '최고기온' 컬럼이 없습니다.")
except (FileNotFoundError, ValueError) as e:
    print(f"데이터 로드 중 예외 발생({e}). 더미 최고기온 데이터를 생성합니다.")
    locs = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주',
            '양구', '포천', '속초', '옹진', '가평', '홍천', '강릉', '동해', '수원', '이천', '원주', '정선', '태백',
            '아산', '천안', '청주', '충주', '영주', '영양', '보령', '공주', '옥천', '상주', '안동', '청송', '포항',
            '익산', '무주', '성주', '부안', '정읍', '합천', '밀양', '울주', '나주', '화순', '순천', '사천', '창원', '해남', '장흥', '여수']
    data = []
    for m in range(1, 13):
        for loc in locs:
            # 월별 평균 기온 시뮬레이션 (겨울: 영하/영상 초반, 여름: 30도 육박)
            if m in [12, 1, 2]:
                base_temp = 2 + np.random.rand() * 5 # 2~7도
            elif m in [6, 7, 8]:
                base_temp = 28 + np.random.rand() * 7 # 28~35도
            else:
                base_temp = 15 + np.random.rand() * 10 # 15~25도
            
            # 위도/지역 특성 반영 (남쪽일수록 따뜻하게, 강원도는 춥게)
            if '강원' in loc or loc in ['양구', '태백', '정선']:
                base_temp -= 3
            if '전남' in loc or '제주' in loc or loc in ['부산', '창원']:
                base_temp += 2
                
            data.append({'월': m, '지점명': loc, '최고기온': round(base_temp, 1), '시군구': loc + '시'})
    df = pd.DataFrame(data)

# 2. 격자 좌표 정의 (기존과 동일)
grid_mapping = [
    [(None, None), (None, None), (None, None), ('양구', '양구군'), ('양구', '양구군'),  (None, None), (None, None)],
    [(None, None), ('포천', '포천시'), ('포천', '포천시'), ('양구', '양구군'), ('고성', '고성군'), (None, None), (None, None)],
    [('옹진', '옹진군'), ('양주', '양주시'), ('가평', '가평군'), ('홍천', '홍천군'), ('강릉', '강릉시'), ('동해', '동해시'), (None, None)],
    [('옹진', '옹진군'), ('수원', '수원시'), ('이천', '이천시'), ('원주', '원주시'), ('정선', '정선군'), ('태백', '태백시'), (None, None)],
    [('아산', '아산시'), ('천안', '천안시'), ('청주', '청주시'), ('충주', '충주시'), ('영주', '영주시'), ('영양', '영양군'),(None, None)],
    [('보령', '보령시'), ('공주', '공주시'), ('옥천', '옥천군'), ('상주', '상주시'), ('안동', '안동시'), ('청송', '청송군'), ('포항', '포항시')],
    [('익산', '익산시'), ('익산', '익산시'), ('무주', '무주군'), ('성주', '성주군'), ('대구', '달성군'), ('포항', '포항시'), ('포항', '포항시')],
    [('부안', '부안군'), ('정읍', '정읍시'), ('무주', '무주군'), ('합천', '합천군'), ('밀양', '밀양시'), ('울주', '울주군'), ('울주', '울주군')],
    [('나주', '나주시'), ('화순', '화순군'), ('순천', '순천시'), ('사천', '사천시'), ('창원', '창원시'), ('창원', '창원시'),(None, None)],
    [('해남', '해남군'), ('장흥', '장흥군'), ('여수', '여수시'), ('사천', '사천시'), (None, None),(None, None), (None, None)]
]

cell_size = 50000
start_x = 900000
start_y = 1600000

grid_points = []
for row_idx, row in enumerate(grid_mapping):
    for col_idx, cell in enumerate(row):
        if cell[0] is not None:
            center_x = start_x + col_idx * cell_size + cell_size / 2
            center_y = start_y + (len(grid_mapping) - 1 - row_idx) * cell_size + cell_size / 2
            grid_points.append({
                'row': row_idx, 'col': col_idx,
                'center_x': center_x, 'center_y': center_y,
                'name': cell[0], 'sigungu': cell[1]
            })
grid_df = pd.DataFrame(grid_points)

def match_location(row):
    sigungu = row['시군구'].strip() if isinstance(row['시군구'], str) else ''
    if sigungu == '포항시': return '포항'
    elif sigungu == '달성군': return '대구'
    elif sigungu == '기장군': return '부산'
    else: return sigungu.replace('시', '').replace('군', '')

df['matched_name'] = df.apply(match_location, axis=1)

# ============================================================================
# 월별 최고기온 히트맵 시각화 (군집화 X, 실제 값 표시)
# ============================================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 18))
fig.suptitle('월별 최고기온 현황 (단위: ℃)', fontsize=22, fontweight='bold')

# 색상맵 설정 (기온은 보통 Coolwarm, RdYlBu 등을 사용)
# 여기서는 낮은 온도(파랑) -> 높은 온도(빨강)인 'coolwarm' 사용
cmap = plt.cm.coolwarm 

monthly_stats = {}

for month in range(1, 13):
    ax = axes[(month-1)//4, (month-1)%4]
    
    # 1. 해당 월 데이터 추출
    month_data = df[df['월'] == month].copy()
    
    # 해당 월의 데이터 매칭
    merged_data = []
    for _, grid_point in grid_df.iterrows():
        matched = month_data[month_data['matched_name'] == grid_point['name']]
        if len(matched) > 0:
            val = matched.iloc[0]['최고기온']
            merged_data.append({
                'value': val,
                'row': grid_point['row'],
                'col': grid_point['col'],
                'center_x': grid_point['center_x'],
                'center_y': grid_point['center_y']
            })
            
    merged_df = pd.DataFrame(merged_data)
    
    if len(merged_df) > 0:
        vmin = merged_df['value'].min()
        vmax = merged_df['value'].max()
        monthly_stats[month] = (vmin, vmax)
    else:
        vmin, vmax = 0, 0
        monthly_stats[month] = (0, 0)

    # 2. 배경 그리기
    korea_outline_x = [start_x, start_x + 7*cell_size, start_x + 7*cell_size, start_x, start_x]
    korea_outline_y = [start_y, start_y, start_y + 10*cell_size, start_y + 10*cell_size, start_y]
    ax.plot(korea_outline_x, korea_outline_y, 'k-', linewidth=1.5, alpha=0.3)
    
    for row_idx in range(len(grid_mapping) + 1):
        y = start_y + row_idx * cell_size
        ax.plot([start_x, start_x + 7*cell_size], [y, y], color='gray', linewidth=0.5, alpha=0.2)
    for col_idx in range(8):
        x = start_x + col_idx * cell_size
        ax.plot([x, x], [start_y, start_y + 10*cell_size], color='gray', linewidth=0.5, alpha=0.2)

    # 3. 각 셀 그리기 (값에 따른 색상 + 텍스트 표시)
    for _, item in merged_df.iterrows():
        val = item['value']
        row = item['row']
        col = item['col']
        
        # 정규화 (해당 월의 최소~최대 기준으로 색상 결정)
        # 편차를 잘 보여주기 위해 월별 Min-Max Scaling 사용
        if vmax - vmin == 0:
            norm_val = 0.5
        else:
            norm_val = (val - vmin) / (vmax - vmin)
            
        color = cmap(norm_val)
        
        rect_x = start_x + col * cell_size
        rect_y = start_y + (len(grid_mapping) - 1 - row) * cell_size
        
        rect = Rectangle((rect_x, rect_y), cell_size, cell_size,
                         facecolor=color, alpha=0.9, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)
        
        # [핵심 요구사항] 각 셀에 값(온도) 텍스트로 표시
        # 글자색: 배경이 너무 어두우면(파랑/빨강 끝자락) 흰색, 중간이면 검은색
        text_color = 'white' if (norm_val < 0.3 or norm_val > 0.7) else 'black'
        
        ax.text(item['center_x'], item['center_y'], f"{val:.1f}", 
                ha='center', va='center', fontsize=9, fontweight='bold', color=text_color)

    ax.set_xlim(start_x - 10000, start_x + 7*cell_size + 10000)
    ax.set_ylim(start_y - 10000, start_y + 10*cell_size + 10000)
    ax.set_aspect('equal')
    ax.set_title(f'{month}월', fontsize=16, fontweight='bold')
    ax.axis('off')

# ============================================================================
# 범례 및 정보 표시
# ============================================================================
# 컬러바 추가 (상대적 온도 분포)
cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02]) 
norm = plt.Normalize(vmin=0, vmax=1)
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
cb.set_label('월별 상대 온도 (파랑: 해당 월 최저 / 빨강: 해당 월 최고)', fontsize=12)
cb.set_ticks([0, 0.5, 1])
cb.set_ticklabels(['Low', 'Avg', 'High'])

# 월별 온도 범위 텍스트 출력
plt.figtext(0.5, 0.045, "※ 각 월별 실제 최고기온 범위 (Min ~ Max)", ha="center", fontsize=12, fontweight='bold')

stats_text_1 = "   ".join([f"[{m}월: {monthly_stats[m][0]:.1f}°C ~ {monthly_stats[m][1]:.1f}°C]" for m in range(1, 7)])
stats_text_2 = "   ".join([f"[{m}월: {monthly_stats[m][0]:.1f}°C ~ {monthly_stats[m][1]:.1f}°C]" for m in range(7, 13)])

plt.figtext(0.5, 0.03, stats_text_1, ha="center", fontsize=10, fontfamily='Monospace')
plt.figtext(0.5, 0.015, stats_text_2, ha="center", fontsize=10, fontfamily='Monospace')

plt.subplots_adjust(bottom=0.15, top=0.95)
plt.savefig('monthly_max_temperature_map.png', dpi=300, bbox_inches='tight')
plt.show()