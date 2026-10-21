// Specialized FSM state for the HandPosition task on G1-29DoF.
// Only a subset of joints (linker Arm) is controlled by the policy.
// Other joints bleiben in der FixStand-Pose, und der Übergang wird weich geblendet.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_HandPosition : public FSMState
{
public:
    State_HandPosition(int state_mode, std::string state_string);
    
    void enter()
    {
        // Warmup-Zähler zurücksetzen (sanfter Übergang bei jedem Eintritt)
        warmup_counter = 0;

        // FixStand-Positionen aus config.yaml laden (statt aus lowstate),
        // damit alle Nicht-Policy-Joints (z.B. Beine, rechter Arm) in dieser Pose bleiben.
        saved_joint_positions.clear();
        try {
            auto fixstand_qs =
                param::config["FSM"]["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
            if (!fixstand_qs.empty() && fixstand_qs.size() > 1)
            {
                // Nehme die letzte Pose aus FixStand (Ziel-Pose)
                saved_joint_positions = fixstand_qs.back();
            }
        } catch (const std::exception& e) {
            // Fallback: aktuelle Positionen aus lowstate
            if (lowstate)
            {
                auto& motors = lowstate->msg_.motor_state();
                if (motors.size() > 0)
                {
                    saved_joint_positions.resize(motors.size());
                    for (size_t i = 0; i < motors.size(); ++i)
                    {
                        saved_joint_positions[i] = motors[i].q();
                    }
                }
            }
        }

        // Policy-Joint-IDs ermitteln (die vom Hand-Policy-Deploy angegebenen DOFs)
        policy_joint_ids.clear();
        YAML::Node actions_cfg = env->cfg["actions"];
        if (actions_cfg["JointPositionAction"]
            && !actions_cfg["JointPositionAction"]["joint_ids"].IsNull())
        {
            policy_joint_ids =
                actions_cfg["JointPositionAction"]["joint_ids"].as<std::vector<int>>();
        }

        // PD-Gains setzen: RL-DOF-Indizes über joint_ids_map auf SDK-Motor-Indizes mappen
        auto& joint_ids_map = env->robot->data.joint_ids_map;
        for (int i = 0; i < env->robot->data.joint_stiffness.size()
                         && i < static_cast<int>(joint_ids_map.size()); ++i)
        {
            int sdk_idx = static_cast<int>(joint_ids_map[i]);
            if (sdk_idx >= 0
                && sdk_idx < static_cast<int>(lowcmd->msg_.motor_cmd().size()))
            {
                lowcmd->msg_.motor_cmd()[sdk_idx].kp() =
                    env->robot->data.joint_stiffness[i];
                lowcmd->msg_.motor_cmd()[sdk_idx].kd() =
                    env->robot->data.joint_damping[i];
                lowcmd->msg_.motor_cmd()[sdk_idx].dq() = 0;
                lowcmd->msg_.motor_cmd()[sdk_idx].tau() = 0;
            }
        }

        // FixStand-Positionen sofort in q setzen, bevor die Policy losläuft
        for (size_t i = 0; i < saved_joint_positions.size()
                        && i < static_cast<size_t>(lowcmd->msg_.motor_cmd().size()); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = saved_joint_positions[i];
        }

        env->robot->update();

        // Policy-Thread starten (wie in State_RLBase)
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // Gespeicherte Joint-Positionen (FixStand) für alle Joints
    std::vector<float> saved_joint_positions;
    // Nur diese DOFs werden von der Policy gesteuert (RL-Indices)
    std::vector<int> policy_joint_ids;

    // Sanfter Übergang von FixStand zur Policy-Pose (Warmup)
    int warmup_steps = 100;
    int warmup_counter = 0;
};

REGISTER_FSM(State_HandPosition)

