.PHONY: all
all: leetcode

CXX = g++
CXXFLAGS = -O3
RSFLAGS = -C "opt-level=0"

mkdir:
	mkdir -p build

build/c: main.c
	cc -O0 -o $@ $< -lm

build/rs: main.rs
	rustc $(RSFLAGS) -o $@ -g $<

build/cpp: main.cpp
	$(CXX) -std=c++26 -O0 -o $@ $< -lm

leetcode: build/cpp build/c build/rs

benches: build/bench_cpp build/bench_rs

save_benches: benches
	./build/bench_cpp > bench.cpp.txt
	./target/release/bench > bench.rs.txt

build/bench_cpp: bench.cpp
	$(CXX) $(CXXFLAGS) -std=c++26 -w -o $@ $<

build/bench_rs: bench.rs
	RUSTFLAGS="-Z threads=16" cargo build --release

clean:
	rm -f build/*
