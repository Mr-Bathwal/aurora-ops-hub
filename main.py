import warnings
warnings.filterwarnings("ignore")   # hides the cosmetic deprecation warnings for a clean menu

from agents import run_system_health, run_log_analyzer, run_backup_dr

from orchestrator import run_orchestrator

from auto_ops import run_auto_ops

def show_menu():
    print("\n" + "=" * 48)
    print("            IT OPS ASSISTANT HUB")
    print("=" * 48)
    print("Choose an agent:")
    print("  1. System Health Agent   - check CPU, memory, disk")
    print("  2. Log Analyzer Agent    - summarize a log file")
    print("  3. Backup & DR Agent     - back up data, check DR status")
    print("  4. Smart Mode            - just describe what you need")
    print("  5. Auto Diagnose & Fix - find and remediate problems automatically")

def main():
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            print("\n[ System Health Agent running... ]\n")
            print(run_system_health())
        elif choice == "2":
            print("\n[ Log Analyzer Agent running... ]\n")
            print(run_log_analyzer())
        elif choice == "3":
            print("\n[ Backup & DR Agent running... ]\n")
            print(run_backup_dr())
        elif choice == "4":
            request = input("\nWhat do you need? ").strip()
            print("\n[ Orchestrator routing your request... ]\n")
            print(run_orchestrator(request))
        elif choice == "5":
            print("\n[ Auto Diagnose & Fix running... ]\n")
            print(run_auto_ops())
        else:
            print("Invalid choice — please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()