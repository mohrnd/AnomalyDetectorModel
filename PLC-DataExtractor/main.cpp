#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <ctime>
#include <modbus/modbus.h>
#include <map>
#include <csignal>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <memory>
#include <algorithm>
#include <filesystem>

using namespace std;

// Global flag for graceful shutdown
atomic<bool> running(true);

struct Variable {
    string name;
    int address;
    uint8_t last_value = 0;
    bool value_changed = true;  // Force initial write

    Variable(const string& n, int addr) : name(n), address(addr) {}
};

class ModbusLogger {
private:
    modbus_t* ctx;
    vector<Variable> variables;
    ofstream output_file;
    string server_ip;
    int server_port;
    int poll_interval_ms;
    int reconnect_attempts;
    int max_reconnect_attempts;
    string output_filename;
    int samples_logged;

    // Performance optimization: batch read addresses
    struct AddressRange {
        int start_addr;
        int count;
        vector<int> var_indices;  // Which variables are in this range
    };
    vector<AddressRange> address_ranges;

public:
    ModbusLogger(const string& ip, int port = 502, int poll_ms = 10)
        : ctx(nullptr), server_ip(ip), server_port(port),
        poll_interval_ms(poll_ms), reconnect_attempts(0), max_reconnect_attempts(5), samples_logged(0) {

        // Initialize variables
        variables = {
            {"A1_Exported", 1}, {"A2_Exported", 2}, {"A3_Exported", 3}, {"A4_Exported", 4},
            {"A5_Exported", 5}, {"A6_Exported", 6}, {"EmergencyStop_Exported", 7},
            {"greenLight_Exported", 8}, {"orangeLight_Exported", 9}, {"P1_2_Armed_Exported", 10},
            {"P1_BallDropCheck_Exported", 11}, {"P1_CartPresence_Exported", 12},
            {"P1_ClockwiseRotation_Exported", 13}, {"P1_CClockwiseRotation_Exported", 14},
            {"P2_CartPresence_Exported", 20}, {"P3_4_Armed_Exported", 23},
            {"P3_BallDropCheck_Exported", 24}, {"P3_CartPresence_Exported", 25},
            {"P3_ClockwiseRotation_Exported", 26}, {"P3_CClockwiseRotation_Exported", 27},
            {"P4_CartPresence_Exported", 35}, {"P5_CartPresence_Exported", 36},
            {"P6_CartPresence_Exported", 37}, {"redLight_Exported", 38},
            {"isConveyorRunning_Exported", 39}, {"runConveyor_Exported", 40},
            {"runSwitch_Exported", 41}
        };

        optimize_address_ranges();
    }

    ~ModbusLogger() {
        graceful_shutdown();
    }

    bool initialize() {
        if (!connect_modbus()) {
            return false;
        }

        if (!open_output_file()) {
            return false;
        }

        write_header();
        return true;
    }

    void run() {
        cout << "Waiting for runSwitch_Exported to be ON..." << endl;

        if (!wait_for_start_condition()) {
            cerr << "Failed to detect start condition" << endl;
            return;
        }

        cout << "runSwitch_Exported is ON. Starting data logging..." << endl;
        cout << "Logging to: " << output_filename << endl;
        cout << "Real-time updates every " << poll_interval_ms << "ms" << endl;
        cout << "Press Ctrl+C to stop gracefully" << endl;
        cout << string(60, '-') << endl;

        auto last_status_time = chrono::steady_clock::now();

        while (running) {
            auto start_time = chrono::high_resolution_clock::now();

            if (log_data_sample()) {
                samples_logged++;

                // Print status every 5 seconds
                auto now = chrono::steady_clock::now();
                if (chrono::duration_cast<chrono::seconds>(now - last_status_time).count() >= 5) {
                    print_status();
                    last_status_time = now;
                }
            } else {
                // Handle reconnection
                if (!handle_connection_error()) {
                    break;
                }
            }

            // Precise timing control
            auto end_time = chrono::high_resolution_clock::now();
            auto elapsed = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
            auto sleep_time = max(0L, static_cast<long>(poll_interval_ms - elapsed));

            if (sleep_time > 0) {
                this_thread::sleep_for(chrono::milliseconds(sleep_time));
            }
        }

        cout << "\n Performing graceful shutdown..." << endl;
        graceful_shutdown();
        cout << "Logging stopped. Total samples: " << samples_logged << endl;
        cout << "Data saved to: " << filesystem::absolute(output_filename) << endl;
    }

private:
    bool connect_modbus() {
        ctx = modbus_new_tcp(server_ip.c_str(), server_port);
        if (!ctx) {
            cerr << "Unable to allocate libmodbus context" << endl;
            return false;
        }

        // Set timeout values
        modbus_set_response_timeout(ctx, 1, 0);  // 1 second timeout
        modbus_set_byte_timeout(ctx, 1, 0);

        if (modbus_connect(ctx) == -1) {
            cerr << "Connection failed: " << modbus_strerror(errno) << endl;
            modbus_free(ctx);
            ctx = nullptr;
            return false;
        }

        cout << "Modbus connection established to " << server_ip << ":" << server_port << endl;
        reconnect_attempts = 0;
        return true;
    }

    bool open_output_file() {
        // Create filename with timestamp
        auto now = chrono::system_clock::now();
        auto time_t = chrono::system_clock::to_time_t(now);
        stringstream ss;
        ss << "modbus_log_" << put_time(localtime(&time_t), "%Y%m%d_%H%M%S") << ".csv";
        output_filename = ss.str();

        // Get absolute path for display
        string abs_path = filesystem::absolute(output_filename);
        cout << "Output file: " << abs_path << endl;
        cout << "Working directory: " << filesystem::current_path() << endl;

        output_file.open(output_filename, ios::out | ios::trunc);
        if (!output_file.is_open()) {
            cerr << "Failed to open output file: " << output_filename << endl;
            return false;
        }

        return true;
    }

    void write_header() {
        output_file << "timestamp_ms,timestamp_iso";
        for (const auto& var : variables) {
            output_file << "," << var.name;
        }
        output_file << "\n";
        output_file.flush();
        cout << "CSV header written" << endl;
    }

    bool wait_for_start_condition() {
        const int max_wait_seconds = 300;  // 5 minutes timeout
        auto start_wait = chrono::steady_clock::now();
        int dots = 0;

        while (running) {
            uint8_t switch_val;
            int rc = modbus_read_bits(ctx, 41, 1, &switch_val);

            if (rc == -1) {
                cerr << "Failed to read runSwitch_Exported: " << modbus_strerror(errno) << endl;
                return false;
            }

            if (switch_val == 1) {
                cout << endl;
                return true;
            }

            // Visual feedback while waiting
            if (dots % 10 == 0) {
                cout << ".";
                cout.flush();
            }
            dots++;

            // Check for timeout
            auto elapsed = chrono::duration_cast<chrono::seconds>(
                               chrono::steady_clock::now() - start_wait).count();
            if (elapsed > max_wait_seconds) {
                cerr << "\n Timeout waiting for start condition" << endl;
                return false;
            }

            this_thread::sleep_for(chrono::milliseconds(100));
        }

        return false;
    }

    bool log_data_sample() {
        uint64_t timestamp_ms = get_timestamp_ms();
        string timestamp_iso = get_iso_timestamp();

        vector<uint8_t> values(variables.size());

        // Read all variables efficiently using batch reads
        if (!read_all_variables(values)) {
            return false;
        }

        // Update last values
        for (size_t i = 0; i < variables.size(); ++i) {
            if (values[i] != variables[i].last_value) {
                variables[i].last_value = values[i];
                variables[i].value_changed = true;
            }
        }

        // Write data to CSV
        output_file << timestamp_ms << "," << timestamp_iso;
        for (const auto& value : values) {
            output_file << "," << static_cast<int>(value);
        }
        output_file << "\n";
        output_file.flush();  // Ensure real-time writing

        return true;
    }

    bool read_all_variables(vector<uint8_t>& values) {
        // Use optimized batch reads when possible
        for (const auto& range : address_ranges) {
            vector<uint8_t> batch_values(range.count);

            int rc = modbus_read_bits(ctx, range.start_addr, range.count, batch_values.data());
            if (rc == -1) {
                cerr << "Failed to read address range " << range.start_addr
                     << "-" << (range.start_addr + range.count - 1)
                     << ": " << modbus_strerror(errno) << endl;
                return false;
            }

            // Map batch results to individual variables
            for (size_t i = 0; i < range.var_indices.size(); ++i) {
                int var_idx = range.var_indices[i];
                int addr_offset = variables[var_idx].address - range.start_addr;
                values[var_idx] = batch_values[addr_offset];
            }
        }

        return true;
    }

    void optimize_address_ranges() {
        // Group consecutive addresses for batch reading
        if (variables.empty()) return;

        // Sort variables by address
        vector<pair<int, int>> addr_to_idx;
        for (size_t i = 0; i < variables.size(); ++i) {
            addr_to_idx.push_back({variables[i].address, i});
        }
        sort(addr_to_idx.begin(), addr_to_idx.end());

        // Create ranges for batch reading
        const int max_gap = 5;  // Maximum gap to still group addresses
        AddressRange current_range;
        current_range.start_addr = addr_to_idx[0].first;
        current_range.var_indices.push_back(addr_to_idx[0].second);

        for (size_t i = 1; i < addr_to_idx.size(); ++i) {
            int addr = addr_to_idx[i].first;
            int var_idx = addr_to_idx[i].second;

            if (addr - current_range.start_addr < max_gap + current_range.var_indices.size()) {
                current_range.var_indices.push_back(var_idx);
            } else {
                // Finish current range and start new one
                current_range.count = current_range.var_indices.size() +
                                      (current_range.var_indices.empty() ? 0 :
                                           variables[current_range.var_indices.back()].address - current_range.start_addr);
                address_ranges.push_back(current_range);

                current_range = AddressRange();
                current_range.start_addr = addr;
                current_range.var_indices.push_back(var_idx);
            }
        }

        // Add the last range
        current_range.count = current_range.var_indices.size() +
                              (current_range.var_indices.empty() ? 0 :
                                   variables[current_range.var_indices.back()].address - current_range.start_addr);
        address_ranges.push_back(current_range);
    }

    bool handle_connection_error() {
        reconnect_attempts++;
        cout << "Connection error. Reconnection attempt " << reconnect_attempts
             << "/" << max_reconnect_attempts << endl;

        if (reconnect_attempts > max_reconnect_attempts) {
            cerr << "Max reconnection attempts reached. Stopping." << endl;
            return false;
        }

        cleanup_connection();
        this_thread::sleep_for(chrono::seconds(2));

        return connect_modbus();
    }

    void print_status() {
        auto now = chrono::system_clock::now();
        auto time_t = chrono::system_clock::to_time_t(now);

        cout << "[" << put_time(localtime(&time_t), "%H:%M:%S") << "] "
             << "Samples: " << samples_logged
             << " | Rate: " << (1000.0 / poll_interval_ms) << " Hz"
             << " | File: " << output_filename << endl;
    }

    uint64_t get_timestamp_ms() {
        using namespace chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }

    string get_iso_timestamp() {
        auto now = chrono::system_clock::now();
        auto time_t = chrono::system_clock::to_time_t(now);
        auto ms = chrono::duration_cast<chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;

        stringstream ss;
        ss << put_time(gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
        ss << "." << setfill('0') << setw(3) << ms.count() << "Z";
        return ss.str();
    }

    void cleanup_connection() {
        if (ctx) {
            modbus_close(ctx);
            modbus_free(ctx);
            ctx = nullptr;
        }
    }

    void graceful_shutdown() {
        cout << "Flushing data to file..." << endl;

        if (output_file.is_open()) {
            output_file.flush();
            output_file.close();
            cout << "File closed successfully" << endl;
        }

        cleanup_connection();
        cout << "Modbus connection closed" << endl;
    }
};

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    cout << "\n Received signal " << signal << ". Initiating graceful shutdown..." << endl;
    running = false;
}

int main(int argc, char* argv[]) {
    // Install signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Parse command line arguments
    string server_ip = "192.168.1.4";
    int server_port = 502;
    int poll_interval = 5;

    if (argc >= 2) server_ip = argv[1];
    if (argc >= 3) server_port = stoi(argv[2]);
    if (argc >= 4) poll_interval = stoi(argv[3]);

    cout << "Server: " << server_ip << ":" << server_port << endl;
    cout << "Poll interval: " << poll_interval << "ms" << endl;
    cout << string(60, '=') << endl;

    try {
        ModbusLogger logger(server_ip, server_port, poll_interval);

        if (!logger.initialize()) {
            cerr << "Failed to initialize logger" << endl;
            return -1;
        }

        logger.run();

    } catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
        return -1;
    }

    cout << "Goodbye!" << endl;
    return 0;
}
