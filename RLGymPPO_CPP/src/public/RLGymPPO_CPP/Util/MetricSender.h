#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>

namespace RLGPC {
    struct RG_IMEXPORT MetricSender {
        std::string curRunID;
        std::string projectName, groupName, runName;
        pybind11::module pyMod;

        MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});

        // Interdire la copie et le déplacement
        MetricSender(const MetricSender&) = delete;
        MetricSender& operator=(const MetricSender&) = delete;
        MetricSender(MetricSender&&) = delete;
        MetricSender& operator=(MetricSender&&) = delete;

        void Send(const Report& report);

        ~MetricSender();
    };
}
