#include "FSM/State_HandPosition.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_HandPosition::State_HandPosition(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_HandPosition::run()
{
    auto action = env->action_manager->processed_actions();

    auto& joint_ids_map = env->robot->data.joint_ids_map;

    // Sanfter Übergang (Warmup) von FixStand-Position zur Policy-Zielpose
    float alpha = 1.0f;
    if (warmup_counter < warmup_steps)
    {
        alpha = static_cast<float>(warmup_counter) / static_cast<float>(warmup_steps);
        warmup_counter++;
    }

    if (!policy_joint_ids.empty())
    {
        // Zuerst: Alle Joints bekommen ihre gespeicherten Positionen (aus FixStand)
        for (size_t i = 0; i < saved_joint_positions.size()
                        && i < static_cast<size_t>(lowcmd->msg_.motor_cmd().size()); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = saved_joint_positions[i];
        }
        
        // Dann: Nur Policy-Joints (z.B. linker Arm) bekommen Actions von der Policy
        // MIT Warmup-Blending: q = q_fixstand + alpha * (q_policy - q_fixstand)
        int n = std::min(static_cast<int>(action.size()),
                         static_cast<int>(policy_joint_ids.size()));
        for (int i = 0; i < n; ++i)
        {
            int rl_idx = policy_joint_ids[i];  // RL-DOF-Index
            if (rl_idx < 0 || rl_idx >= static_cast<int>(joint_ids_map.size()))
            {
                continue;
            }
            int sdk_idx = static_cast<int>(joint_ids_map[rl_idx]);  // Mapping RL -> SDK
            if (sdk_idx >= 0 && sdk_idx < static_cast<int>(lowcmd->msg_.motor_cmd().size()))
            {
                float q_policy = action[i];

                float q_fix = 0.0f;
                if (sdk_idx >= 0
                    && sdk_idx < static_cast<int>(saved_joint_positions.size()))
                {
                    q_fix = saved_joint_positions[sdk_idx];
                }

                float q_blend = q_fix + alpha * (q_policy - q_fix);
                lowcmd->msg_.motor_cmd()[sdk_idx].q() = q_blend;
            }
        }
    }
    else
    {
        // Fallback: Standard-Mapping (sollte im Normalfall für Hand-Policy nicht benutzt werden)
        int n = std::min(static_cast<int>(action.size()),
                         static_cast<int>(joint_ids_map.size()));
        for (int i = 0; i < n; ++i)
        {
            lowcmd->msg_.motor_cmd()[joint_ids_map[i]].q() = action[i];
        }
    }
}

