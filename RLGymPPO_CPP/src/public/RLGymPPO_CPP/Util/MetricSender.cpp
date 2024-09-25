#include "MetricSender.h"

#include "Timer.h"

namespace py = pybind11;
using namespace RLGPC;

RLGPC::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID)
    : projectName(_projectName), groupName(_groupName), runName(_runName) {

    RG_LOG("Initializing MetricSender...");

    try {
        pyMod = py::module::import("python_scripts.metric_receiver");
    }
    catch (const std::exception& e) {
        RG_ERR_CLOSE("MetricSender: Failed to import metrics receiver, exception: " << e.what());
    }

    try {
        auto returnedRunID = pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
        curRunID = returnedRunID.cast<std::string>();
        RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

    }
    catch (const std::exception& e) {
        RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
    }

    RG_LOG(" > MetricSender initialized.");
}

void RLGPC::MetricSender::Send(const Report& report) {
    if (!pyMod) {
        RG_ERR_CLOSE("MetricSender: Python module not initialized.");
    }

    py::dict reportDict;

    for (const auto& pair : report.data) {
        reportDict[pair.first.c_str()] = pair.second;
    }

    try {
        pyMod.attr("add_metrics")(reportDict);
    }
    catch (const std::exception& e) {
        RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
    }
}

RLGPC::MetricSender::~MetricSender() {
    // Nettoyage si nécessaire
}
