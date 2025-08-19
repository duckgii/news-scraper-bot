# run_all.py
import subprocess, sys

def run(cmd):
    print(f"\n▶ {cmd}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

if __name__ == "__main__":
    run("python naver_econ_news.py")
    run("python push_notion_daily.py")
    print("\n✅ 완료")