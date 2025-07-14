import os
import itertools

# 폴더 생성
os.makedirs("configs_list", exist_ok=True)

# 항목 정의
feature_extractors = ["BRIEF", "SIFT", "ORB", "FAST", "Harris", "ShiTomasi"]
feature_matchers = ["BFMatcher", "FLANNMatcher", "LoFTR", "GlueStick", "LightGlue"]

# Transformation Estimator는 고정
transform_estimators = ["SVDTF"]


# 설정 생성 함수
def generate_config_lines(selected_extractor, selected_matcher):
    lines = []

    # Feature Extractors
    lines.append("# Feature Extractors")
    for feat in feature_extractors:
        val = "True" if feat == selected_extractor else "False"
        lines.append(f"{feat} = {val}")
    lines.append("")

    # Feature Matchers
    lines.append("# Feature Matchers")
    for matcher in feature_matchers:
        val = "True" if matcher == selected_matcher else "False"
        lines.append(f"{matcher} = {val}")
    lines.append("")

    # Transformation Estimator (고정)
    lines.append("# Transformation Estimator")
    for est in transform_estimators:
        lines.append(f"{est} = True")

    return lines


# 조합 생성 및 파일 저장
index = 1
for extractor, matcher in itertools.product(feature_extractors, feature_matchers):
    lines = generate_config_lines(extractor, matcher)
    filename = f"configs_list/config_{index:02d}.yaml"
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    index += 1
