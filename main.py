# Copyright 2022 Cristian Grosu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from hierarchical_flow import hierarchical_flow, hierarchical_flow_Q4
from simple_flow import simple_flow

QUESTION = "Q1"

def main():
    question_solvers = {
        "Q1": simple_flow,
        "Q2": simple_flow,
        "Q3_A": hierarchical_flow,
        "Q3_B": hierarchical_flow,
        "Q4_A": hierarchical_flow_Q4,
        "Q4_B": hierarchical_flow_Q4
    }
    if QUESTION in question_solvers:
        question_solvers[QUESTION](QUESTION)
    else:
        print("Question not implemented")

if __name__ == "__main__":
    main()