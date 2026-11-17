// Native ns-3 UAV task-offloading episode runner.
//
// This version uses ns-3 Mobility, Wi-Fi device/channel setup, and the
// LogDistancePropagationLossModel. Project-specific glue provides task
// scheduling, queue/memory accounting, radio/CPU energy accounting, and the
// synchronous stdin/stdout RL bridge. The current version remains a
// link-budget/task-level model: it does not transmit UDP application packets.

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/propagation-module.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace ns3;

namespace
{
constexpr uint32_t N_USERS = 10;
constexpr uint32_t N_EDGES = 3;
constexpr uint32_t N_DEST = 5;     // Local, 3 edge UAVs, cloud.
constexpr uint32_t N_ACTIONS = 16; // 15 unordered destination pairs + DROP.
constexpr double LOG_DISTANCE_REFERENCE_LOSS_DB = 40.05; // Friis at 1 m, 2.4 GHz.

struct NodeState
{
    double cpuHz = 0.0;
    double batteryJ = 0.0;
    double memoryTotalMb = 0.0;
    double memoryUsedMb = 0.0;
    double cpuAvailableAtS = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Episode
{
    uint32_t seed = 7;
    double duration = 60.0;
    double arrival = 1.5;
    double cycles = 2e9;
    double deadline = 2.0;
    double inputMb = 1.0;
    double outputMb = 0.1;
    double taskMemoryMb = 128.0;
    double userCpu = 5e8;
    double edgeCpu = 2e9;
    double cloudCpu = 2e10;
    double userBattery = 1000.0;
    double edgeBattery = 1500.0;
    double cloudBattery = 1e12;
    double userMemory = 512.0;
    double edgeMemory = 2048.0;
    double cloudMemory = 32768.0;
    double txW = 5.0;
    double rxW = 3.0;
    double bwMbps = 6.0;
    double wifiRateMbps = 6.0;
    double minLinkRateMbps = 0.5;
    double cpuCoeff = 1e-29;
    double pathLossExponent = 2.8;
    double shadowingStdDb = 4.0;
    double dropLatencyS = 10.0;
    double latencyRefS = 1.0;
    double energyRefJ = 10.0;
    double imbalanceRef = 1.0;
    double latencyWeight = 0.60;
    double energyWeight = 0.20;
    double imbalanceWeight = 0.20;
};

Episode g;
std::vector<NodeState> nodes;
std::vector<Ptr<MobilityModel>> mobilityModels;
Ptr<LogDistancePropagationLossModel> lossModel;
std::vector<std::vector<double>> shadowingDb;

uint32_t taskNo = 0;
uint32_t totalTasks = 0;
std::mt19937 gRng;
double sumLatency = 0.0;
double sumEnergy = 0.0;
double sumObjective = 0.0;
uint32_t success = 0;

std::vector<std::pair<int, int>> pairs;

void
InitPairs()
{
    pairs.clear();
    for (uint32_t i = 0; i < N_DEST; ++i)
    {
        for (uint32_t j = i; j < N_DEST; ++j)
        {
            pairs.emplace_back(static_cast<int>(i), static_cast<int>(j));
        }
    }
}

uint32_t
DestinationNode(uint32_t candidate, uint32_t user)
{
    if (candidate == 0)
    {
        return user;
    }
    if (candidate == N_DEST - 1)
    {
        return N_USERS + N_EDGES;
    }
    return N_USERS + candidate - 1;
}

double
QueueWaitS(uint32_t nodeId, double nowS)
{
    return std::max(0.0, nodes.at(nodeId).cpuAvailableAtS - nowS);
}

void
ReleaseMemory(uint32_t nodeId, double amountMb)
{
    NodeState& node = nodes.at(nodeId);
    node.memoryUsedMb = std::max(0.0, node.memoryUsedMb - amountMb);
}

double
RateMbps(uint32_t src, uint32_t dst)
{
    if (src >= mobilityModels.size() || dst >= mobilityModels.size() || !lossModel)
    {
        return 0.1;
    }

    // Log-distance path loss is calculated by ns-3. A fixed, per-link
    // Gaussian shadowing offset is sampled once per episode (slow shadowing),
    // rather than changing independently on every task observation.
    double rxDbm = lossModel->CalcRxPower(16.0, mobilityModels[src], mobilityModels[dst]);
    if (src < shadowingDb.size() && dst < shadowingDb[src].size())
    {
        rxDbm -= shadowingDb[src][dst];
    }

    // Conservative link-budget-to-effective-rate approximation. This is not
    // measured packet-level MAC throughput; see README implementation status.
    const double snrDb = rxDbm - (-96.0);
    const double spectralEfficiency =
        std::max(0.05, std::min(2.0, std::log2(1.0 + std::pow(10.0, snrDb / 10.0))));
    return std::max(0.1, std::min(g.bwMbps, g.wifiRateMbps * spectralEfficiency / 2.0));
}

double
EncodeDelayFeature(double seconds)
{
    constexpr double maxSeconds = 120.0;
    const double bounded = std::clamp(seconds, 0.0, maxSeconds);
    return std::log1p(bounded) / std::log1p(maxSeconds);
}

void
AskAgent(uint32_t user)
{
    if (taskNo >= totalTasks)
    {
        return;
    }

    const double nowS = Simulator::Now().GetSeconds();
    const double subtaskMemoryMb = g.taskMemoryMb / 2.0;

    // Observation is fixed at 60 values:
    // 5 task/source features + 5 candidates x 9 features + 10 system features.
    // Per-candidate feature 6 is now available-memory fraction (not duplicate CPU).
    std::vector<double> obs(60, 0.0);
    obs[0] = g.inputMb / 16.0;
    obs[1] = g.outputMb / 4.0;
    obs[2] = g.cycles / 4e9;
    obs[3] = g.deadline / 2.0;
    obs[4] = nodes[user].batteryJ / g.userBattery;

    for (uint32_t c = 0; c < N_DEST; ++c)
    {
        const uint32_t idx = DestinationNode(c, user);
        const uint32_t base = 5 + c * 9;
        const NodeState& node = nodes[idx];
        const double rate = (c == 0) ? 1000.0 : RateMbps(user, idx);
        const double availableMemory =
            std::max(0.0, node.memoryTotalMb - node.memoryUsedMb);

        obs[base + 0] = std::clamp(node.cpuHz / g.cloudCpu, 0.0, 1.0);
        obs[base + 1] = EncodeDelayFeature(QueueWaitS(idx, nowS));
        const double batteryCapacity =
            (c == N_DEST - 1) ? g.cloudBattery : ((c == 0) ? g.userBattery : g.edgeBattery);
        obs[base + 2] = std::clamp(node.batteryJ / batteryCapacity, 0.0, 1.0);
        obs[base + 3] = (c == 0) ? 1.0 : std::clamp(rate / std::min(g.wifiRateMbps, g.bwMbps), 0.0, 1.0);
        obs[base + 4] = std::clamp(
            mobilityModels[user]->GetDistanceFrom(mobilityModels[idx]) / 500.0, 0.0, 1.0);
        obs[base + 5] = (c == 0) ? 0.0 : EncodeDelayFeature(g.inputMb * 8.0 / rate);
        obs[base + 6] = (node.memoryTotalMb > 0.0)
                            ? std::clamp(availableMemory / node.memoryTotalMb, 0.0, 1.0)
                            : 0.0;
        obs[base + 7] = (c == N_DEST - 1) ? 1.0 : 0.0;
        obs[base + 8] = (c == 0) ? 1.0 : 0.0;
    }

    // System features: 3 edge queues, 3 edge battery fractions,
    // 3 edge memory-availability fractions, and cloud queue.
    for (uint32_t edge = 0; edge < N_EDGES; ++edge)
    {
        const uint32_t idx = N_USERS + edge;
        obs[50 + edge] = EncodeDelayFeature(QueueWaitS(idx, nowS));
        obs[53 + edge] = std::clamp(nodes[idx].batteryJ / g.edgeBattery, 0.0, 1.0);
        obs[56 + edge] = std::clamp(
            std::max(0.0, nodes[idx].memoryTotalMb - nodes[idx].memoryUsedMb) /
                nodes[idx].memoryTotalMb,
            0.0,
            1.0);
    }
    const uint32_t cloudId = N_USERS + N_EDGES;
    obs[59] = EncodeDelayFeature(QueueWaitS(cloudId, nowS));

    std::vector<int> mask(N_ACTIONS, 1);
    if (nodes[user].batteryJ <= 0.0)
    {
        for (uint32_t a = 0; a < N_ACTIONS - 1; ++a)
        {
            mask[a] = 0;
        }
    }

    // Feasibility includes memory, minimum bidirectional link quality, and
    // enough remaining battery to complete this task. This avoids actions
    // that are physically infeasible while preserving battery depletion as a
    // meaningful long-horizon resource-management problem.
    for (uint32_t a = 0; a < pairs.size(); ++a)
    {
        const auto [x, y] = pairs[a];
        const int destinations[2] = {x, y};
        std::map<uint32_t, double> requiredMemory;
        std::map<uint32_t, double> requiredBattery;
        double sourceRadioBattery = 0.0;

        for (int destination : destinations)
        {
            const uint32_t idx = DestinationNode(static_cast<uint32_t>(destination), user);
            requiredMemory[idx] += subtaskMemoryMb;
            const double cycles = g.cycles / 2.0;
            const double inputMb = g.inputMb / 2.0;
            const double outputMb = g.outputMb / 2.0;
            const double uplink = (destination == 0) ? 0.0 : inputMb * 8.0 / RateMbps(user, idx);
            const double downlink = (destination == 0) ? 0.0 : outputMb * 8.0 / RateMbps(idx, user);
            if (destination != 0 &&
                (RateMbps(user, idx) < g.minLinkRateMbps ||
                 RateMbps(idx, user) < g.minLinkRateMbps))
            {
                mask[a] = 0;
                break;
            }

            const double cpuEnergy = g.cpuCoeff * cycles * nodes[idx].cpuHz * nodes[idx].cpuHz;
            if (destination == 0)
            {
                requiredBattery[idx] += cpuEnergy;
            }
            else
            {
                const double userRadio = g.txW * uplink + g.rxW * downlink;
                const double serverRadio = g.rxW * uplink + g.txW * downlink;
                sourceRadioBattery += userRadio;
                requiredBattery[idx] += cpuEnergy + serverRadio;
            }
        }
        if (!mask[a])
        {
            continue;
        }
        requiredBattery[user] += sourceRadioBattery;

        for (const auto& [id, requiredMb] : requiredMemory)
        {
            const NodeState& node = nodes[id];
            const double availableMb = node.memoryTotalMb - node.memoryUsedMb;
            if (node.batteryJ <= requiredBattery[id] || availableMb + 1e-9 < requiredMb)
            {
                mask[a] = 0;
                break;
            }
        }
        if (mask[a])
        {
            for (const auto& [id, requiredJ] : requiredBattery)
            {
                if (nodes[id].batteryJ <= requiredJ)
                {
                    mask[a] = 0;
                    break;
                }
            }
        }
    }

    // DROP is always feasible.
    std::cout << "OBS";
    for (double value : obs)
    {
        std::cout << ' ' << value;
    }
    std::cout << " MASK";
    for (int value : mask)
    {
        std::cout << ' ' << value;
    }
    std::cout << std::endl;

    std::string line;
    if (!std::getline(std::cin, line))
    {
        Simulator::Stop();
        return;
    }

    std::istringstream inputStream(line);
    std::string tag;
    int action = static_cast<int>(N_ACTIONS - 1);
    inputStream >> tag >> action;
    if (tag != "ACTION" || action < 0 || action >= static_cast<int>(N_ACTIONS) || !mask[action])
    {
        action = static_cast<int>(N_ACTIONS - 1);
    }

    // DROP uses a fixed, explicit latency penalty. It does not depend on the
    // deadline and therefore does not reintroduce a hidden deadline term.
    double latency = g.dropLatencyS;
    double energy = 0.05; // fixed control/drop energy
    double imbalance = 0.0;

    if (action < static_cast<int>(N_ACTIONS - 1))
    {
        const auto [x, y] = pairs[static_cast<size_t>(action)];
        const int destinations[2] = {x, y};
        double finishRelativeS = 0.0;

        for (int destination : destinations)
        {
            const uint32_t idx = DestinationNode(static_cast<uint32_t>(destination), user);
            const double cycles = g.cycles / 2.0;
            const double inputMb = g.inputMb / 2.0;
            const double outputMb = g.outputMb / 2.0;
            const double uplink = (destination == 0) ? 0.0 : inputMb * 8.0 / RateMbps(user, idx);
            const double downlink = (destination == 0) ? 0.0 : outputMb * 8.0 / RateMbps(idx, user);

            // CPU queue is based on ns-3 simulation time, not an arbitrary
            // fixed decrement per task. Upload completes before CPU service;
            // different destinations can process their subtasks in parallel.
            const double serverArrivalS = nowS + uplink;
            const double serviceStartS = std::max(serverArrivalS, nodes[idx].cpuAvailableAtS);
            const double computeS = cycles / nodes[idx].cpuHz;
            const double serviceEndS = serviceStartS + computeS;
            const double subtaskCompletionS = serviceEndS + downlink;
            finishRelativeS = std::max(finishRelativeS, subtaskCompletionS - nowS);
            nodes[idx].cpuAvailableAtS = serviceEndS;

            // Memory is reserved while a subtask is in flight/queued/executing
            // and released at its modeled completion time.
            nodes[idx].memoryUsedMb += subtaskMemoryMb;
            Simulator::Schedule(Seconds(std::max(0.0, subtaskCompletionS - nowS)),
                                &ReleaseMemory,
                                idx,
                                subtaskMemoryMb);

            const double cpuEnergy = g.cpuCoeff * cycles * nodes[idx].cpuHz * nodes[idx].cpuHz;
            const double userRadioEnergy = (destination == 0)
                                               ? 0.0
                                               : (g.txW * uplink + g.rxW * downlink);
            const double serverRadioEnergy = (destination == 0)
                                                 ? 0.0
                                                 : (g.rxW * uplink + g.txW * downlink);

            // Report total system energy: CPU energy plus both radio endpoints.
            energy += cpuEnergy + userRadioEnergy + serverRadioEnergy;
            nodes[idx].batteryJ -= cpuEnergy + serverRadioEnergy;
            nodes[user].batteryJ -= userRadioEnergy;
        }

        latency = finishRelativeS + 0.02;

        double queueMean = 0.0;
        for (uint32_t edge = 0; edge < N_EDGES; ++edge)
        {
            queueMean += QueueWaitS(N_USERS + edge, nowS) / static_cast<double>(N_EDGES);
        }
        double queueVariance = 0.0;
        for (uint32_t edge = 0; edge < N_EDGES; ++edge)
        {
            const double q = QueueWaitS(N_USERS + edge, nowS);
            queueVariance += std::pow(q - queueMean, 2) / static_cast<double>(N_EDGES);
        }
        imbalance = std::min(1.0, std::sqrt(queueVariance) / (queueMean + 0.1));
    }

    const double objective = g.latencyWeight * (latency / g.latencyRefS) +
                             g.energyWeight * (energy / g.energyRefJ) +
                             g.imbalanceWeight * (imbalance / g.imbalanceRef);
    const double reward = -objective;
    sumLatency += latency;
    sumEnergy += energy;
    sumObjective += objective;
    if (latency <= g.deadline)
    {
        ++success;
    }

    ++taskNo;
    const bool done = (taskNo >= totalTasks);
    std::cout << "STEP " << reward << ' ' << latency << ' ' << energy << ' '
              << (latency <= g.deadline ? 1 : 0) << ' ' << (done ? 1 : 0) << std::endl;

    if (done)
    {
        std::cout << "SUMMARY " << totalTasks << ' '
                  << sumLatency / static_cast<double>(totalTasks) << ' '
                  << sumEnergy / static_cast<double>(totalTasks) << ' '
                  << static_cast<double>(success) / static_cast<double>(totalTasks) << ' '
                  << sumObjective / static_cast<double>(totalTasks) << std::endl;
        Simulator::Stop();
    }
}

std::string
WifiModeForRate(double rateMbps)
{
    const int rate = static_cast<int>(std::lround(rateMbps));
    switch (rate)
    {
    case 6:
    case 9:
    case 12:
    case 18:
    case 24:
    case 36:
    case 48:
    case 54:
        return "ErpOfdmRate" + std::to_string(rate) + "Mbps";
    default:
        NS_ABORT_MSG("Unsupported 802.11g PHY rate: " << rateMbps
                                                       << " Mbps. Use 6, 9, 12, 18, 24, 36, 48, or 54.");
    }
    return "ErpOfdmRate6Mbps";
}
} // namespace

int
main(int argc, char** argv)
{
    CommandLine cmd;
    cmd.AddValue("seed", "Random seed", g.seed);
    cmd.AddValue("duration", "Episode duration (s)", g.duration);
    cmd.AddValue("arrival", "Poisson task arrivals per user per second", g.arrival);
    cmd.AddValue("cycles", "CPU cycles per task", g.cycles);
    cmd.AddValue("deadline", "Deadline (evaluation metric only)", g.deadline);
    cmd.AddValue("inputMb", "Task input size (MB)", g.inputMb);
    cmd.AddValue("outputMb", "Task output size (MB)", g.outputMb);
    cmd.AddValue("taskMemoryMb", "Memory requirement per full task (MB)", g.taskMemoryMb);
    cmd.AddValue("txW", "Radio transmit power consumption (W)", g.txW);
    cmd.AddValue("rxW", "Radio receive power consumption (W)", g.rxW);
    cmd.AddValue("bwMbps", "Effective link rate cap (Mbps)", g.bwMbps);
    cmd.AddValue("minLinkRateMbps", "Minimum bidirectional offload rate (Mbps)", g.minLinkRateMbps);
    cmd.AddValue("wifiRateMbps", "Configured 802.11g PHY mode rate (Mbps)", g.wifiRateMbps);
    cmd.AddValue("userCpu", "User CPU capacity (Hz)", g.userCpu);
    cmd.AddValue("edgeCpu", "Edge CPU capacity (Hz)", g.edgeCpu);
    cmd.AddValue("cloudCpu", "Cloud CPU capacity (Hz)", g.cloudCpu);
    cmd.AddValue("userBattery", "User battery capacity (J)", g.userBattery);
    cmd.AddValue("edgeBattery", "Edge battery capacity (J)", g.edgeBattery);
    cmd.AddValue("cloudBattery", "Cloud battery capacity (J)", g.cloudBattery);
    cmd.AddValue("userMemoryMb", "User memory capacity (MB)", g.userMemory);
    cmd.AddValue("edgeMemoryMb", "Edge memory capacity (MB)", g.edgeMemory);
    cmd.AddValue("cloudMemoryMb", "Cloud memory capacity (MB)", g.cloudMemory);
    cmd.AddValue("cpuCoeff", "CPU energy coefficient", g.cpuCoeff);
    cmd.AddValue("pathLossExponent", "Log-distance propagation exponent", g.pathLossExponent);
    cmd.AddValue("shadowingStdDb", "Per-link shadowing standard deviation (dB)", g.shadowingStdDb);
    cmd.AddValue("dropLatencyS", "Fixed effective latency penalty for DROP (s)", g.dropLatencyS);
    cmd.AddValue("latencyRefS", "Latency objective reference (s)", g.latencyRefS);
    cmd.AddValue("energyRefJ", "Energy objective reference (J)", g.energyRefJ);
    cmd.AddValue("imbalanceRef", "Load imbalance objective reference", g.imbalanceRef);
    cmd.AddValue("latencyWeight", "Latency objective weight", g.latencyWeight);
    cmd.AddValue("energyWeight", "Energy objective weight", g.energyWeight);
    cmd.AddValue("imbalanceWeight", "Load imbalance objective weight", g.imbalanceWeight);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(g.duration <= 0.0 || g.arrival < 0.0 || g.cycles <= 0.0 ||
                        g.deadline <= 0.0 || g.userCpu <= 0.0 || g.edgeCpu <= 0.0 ||
                        g.cloudCpu <= 0.0 || g.userBattery <= 0.0 || g.edgeBattery <= 0.0 ||
                        g.cloudBattery <= 0.0 || g.userMemory <= 0.0 || g.edgeMemory <= 0.0 ||
                        g.cloudMemory <= 0.0 || g.taskMemoryMb <= 0.0 || g.pathLossExponent <= 0.0 ||
                        g.minLinkRateMbps <= 0.0 ||
                        g.dropLatencyS <= 0.0 || g.latencyRefS <= 0.0 || g.energyRefJ <= 0.0 ||
                        g.imbalanceRef <= 0.0 || g.shadowingStdDb < 0.0,
                    "Invalid non-positive scenario parameter");
    NS_ABORT_MSG_IF(std::abs(g.latencyWeight + g.energyWeight + g.imbalanceWeight - 1.0) > 1e-9 ||
                        g.latencyWeight < 0.0 || g.energyWeight < 0.0 || g.imbalanceWeight < 0.0,
                    "Objective weights must be non-negative and sum to one");

    RngSeedManager::SetSeed(g.seed);
    RngSeedManager::SetRun(g.seed);
    InitPairs();
    gRng.seed(g.seed);

    std::poisson_distribution<uint32_t> arrivals(g.arrival * N_USERS * g.duration);
    totalTasks = std::max(1u, arrivals(gRng));

    NodeContainer all;
    all.Create(N_USERS + N_EDGES + 1);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    const std::string wifiMode = WifiModeForRate(g.wifiRateMbps);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue(wifiMode),
                                 "ControlMode",
                                 StringValue(wifiMode));

    // The same log-distance path-loss family/exponent is used by the Wi-Fi
    // channel configuration and by the task-level rate estimator.
    YansWifiChannelHelper channel;
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                               "Exponent",
                               DoubleValue(g.pathLossExponent),
                               "ReferenceDistance",
                               DoubleValue(1.0),
                               "ReferenceLoss",
                               DoubleValue(LOG_DISTANCE_REFERENCE_LOSS_DB));

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    wifi.Install(phy, mac, all);

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(all);

    nodes.resize(all.GetN());
    mobilityModels.resize(all.GetN());
    lossModel = CreateObject<LogDistancePropagationLossModel>();
    lossModel->SetAttribute("Exponent", DoubleValue(g.pathLossExponent));
    lossModel->SetAttribute("ReferenceDistance", DoubleValue(1.0));
    lossModel->SetAttribute("ReferenceLoss", DoubleValue(LOG_DISTANCE_REFERENCE_LOSS_DB));

    for (uint32_t i = 0; i < all.GetN(); ++i)
    {
        const double x = (i < N_USERS) ? (i % 5) * 70.0 : ((i - N_USERS + 1) * 110.0);
        const double y = (i < N_USERS) ? (i / 5) * 90.0 : 180.0;
        const double z = (i >= N_USERS && i < N_USERS + N_EDGES) ? 80.0 : 0.0;

        Ptr<MobilityModel> mobilityModel = all.Get(i)->GetObject<MobilityModel>();
        mobilityModel->SetPosition(Vector(x, y, z));
        mobilityModels[i] = mobilityModel;

        const bool isUser = (i < N_USERS);
        const bool isEdge = (i >= N_USERS && i < N_USERS + N_EDGES);
        nodes[i] = {isUser ? g.userCpu : (isEdge ? g.edgeCpu : g.cloudCpu),
                    isUser ? g.userBattery : (isEdge ? g.edgeBattery : g.cloudBattery),
                    isUser ? g.userMemory : (isEdge ? g.edgeMemory : g.cloudMemory),
                    0.0,
                    0.0,
                    x,
                    y,
                    z};
    }

    // Slow per-link shadowing: one zero-mean Gaussian offset per undirected
    // link for this episode, symmetric in both directions.
    shadowingDb.assign(all.GetN(), std::vector<double>(all.GetN(), 0.0));
    if (g.shadowingStdDb > 0.0)
    {
        std::normal_distribution<double> shadowing(0.0, g.shadowingStdDb);
        for (uint32_t i = 0; i < all.GetN(); ++i)
        {
            for (uint32_t j = i + 1; j < all.GetN(); ++j)
            {
                const double offsetDb = shadowing(gRng);
                shadowingDb[i][j] = offsetDb;
                shadowingDb[j][i] = offsetDb;
            }
        }
    }

    std::cout << "READY " << totalTasks << std::endl;

    std::uniform_real_distribution<double> arrivalTime(0.0, g.duration);
    std::uniform_int_distribution<uint32_t> pickUser(0, N_USERS - 1);
    std::vector<std::pair<double, uint32_t>> arrivalsList;
    arrivalsList.reserve(totalTasks);
    for (uint32_t i = 0; i < totalTasks; ++i)
    {
        arrivalsList.emplace_back(arrivalTime(gRng), pickUser(gRng));
    }
    std::sort(arrivalsList.begin(), arrivalsList.end());

    for (const auto& event : arrivalsList)
    {
        Simulator::Schedule(Seconds(event.first), &AskAgent, event.second);
    }

    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
