cmake_minimum_required(VERSION 3.14)
project(ELiteUtils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
# pybind11 서브디렉토리 추가
add_subdirectory(third_party/pybind11)

# include 디렉토리 설정
include_directories(include)

# src 폴더의 모든 .cpp 파일 모으기
file(GLOB CPP_SOURCES src/*.cpp)

# pybind11로 모듈 빌드
pybind11_add_module(elite_utils ${CPP_SOURCES})

# (필요시) 추가 링크 라이브러리나 정의 추가
target_include_directories(elite_utils PRIVATE include)