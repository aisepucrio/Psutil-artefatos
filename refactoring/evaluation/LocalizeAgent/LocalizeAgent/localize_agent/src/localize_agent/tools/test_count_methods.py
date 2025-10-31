from codecarbon import EmissionsTracker
import psutil
import csv
import os
from custom_tools import CountMethods, VariableUsage, FanInFanOutAnalysis, ClassCouplingAnalysis

def analyze_java_code(java_code):
    """Analyzes Java code using various tools."""
    results = {}
    try:
        count_methods_tool = CountMethods()
        varibale_usage_tool = VariableUsage()
        fan_tool = FanInFanOutAnalysis()
        class_coupling_tool = ClassCouplingAnalysis()

        results['count_methods'] = count_methods_tool._run(java_code)
        results['variable_usage'] = varibale_usage_tool._run(java_code)
        results['fan_in_out'] = fan_tool._run(java_code)
        results['class_coupling'] = class_coupling_tool._run(java_code)

    except Exception as e:
        results['error'] = f"Error processing Java source code: {e}"
    return results

def get_java_code():
    """Returns sample Java code as a string."""
    return """
        import java.util.ArrayList;
        import java.util.List;

        package localize_agent.datasets;


        public class InputProcessor {

            private List<String> inputs;
            private String lastProcessedInput;
            private int processCount;

            public InputProcessor() {
                this.inputs = new ArrayList<>();
                this.lastProcessedInput = null;
                this.processCount = 0;
            }

            public void addInput(String input) {
                inputs.add(input);
            }

            public void processInputs() {
                for (String input : inputs) {
                    processInput(input);
                }
            }

            private void processInput(String input) {
                // Simulate processing
                System.out.println("Processing: " + input);
                lastProcessedInput = input;
                processCount++;
            }

            public String getLastProcessedInput() {
                return lastProcessedInput;
            }

            public int getProcessCount() {
                return processCount;
            }

            public void resetProcessor() {
                inputs.clear();
                lastProcessedInput = null;
                processCount = 0;
            }

            public void printAllInputs() {
                System.out.println("All Inputs:");
                for (String input : inputs) {
                    System.out.println(input);
                }
            }

            public void removeInput(String input) {
                inputs.remove(input);
            }

            public void processAndReset() {
                processInputs();
                resetProcessor();
            }

            public void addAndProcess(String input) {
                int a = 1;
                addInput(input);
                processInput(input);
            }

            public void printStatus() {
                System.out.println("Last Processed Input: " + lastProcessedInput);
                System.out.println("Total Process Count: " + processCount);
            }
        }
    """


def test_count_methods():
    """Tests the code analysis tools."""
    java_code = get_java_code()
    results = analyze_java_code(java_code)

    if 'error' in results:
        print(results['error'])
    else:
        print(results['count_methods'])
        print(results['variable_usage'])
        print(results['fan_in_out'])
        print(results['class_coupling'])


def save_psutil_data(mem_start, mem_end, cpu_start, cpu_end):
    """Saves psutil data to a CSV file."""
    csv_file = "psutil_data.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])
        writer.writerow([
            __file__,
            f"{mem_start:.2f}",
            f"{mem_end:.2f}",
            f"{mem_end - mem_start:.2f}",
            f"{cpu_start:.2f}",
            f"{cpu_end:.2f}"
        ])



if __name__ == "__main__":
    tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\LocalizeAgent\localize_agent\src\localize_agent\tools\test_count_methods.py")
    tracker.start()

    mem_start = psutil.virtual_memory().used / (1024**2)
    cpu_start = psutil.cpu_percent(interval=None)

    test_count_methods()

    mem_end = psutil.virtual_memory().used / (1024**2)
    cpu_end = psutil.cpu_percent(interval=None)

    save_psutil_data(mem_start, mem_end, cpu_start, cpu_end)

    tracker.stop()
