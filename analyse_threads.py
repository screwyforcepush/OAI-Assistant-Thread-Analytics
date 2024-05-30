
import dspy
from dotenv import load_dotenv
import os
from get_threads import load_threads_from_file
from ThreadObject import ThreadObject
from dspy.teleprompt import Ensemble
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from collections import Counter
import re
from datetime import datetime
import json
load_dotenv(override=True)
from threads_labeled import labeled_threads
email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

# gpt-4-turbo
# inpput US$10.00 / 1M tokens
# output US$30.00 / 1M tokens

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# Set up the LM
turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=2000)
gpt4o = dspy.OpenAI(model='gpt-4o', max_tokens=2000)

dspy.settings.configure(lm=turbo, trace=[])
dspy.settings.configure(log_openai_usage=True)

objective_path = "objective.json"
threads = load_threads_from_file()


#%%

def get_milestone_names(data: dict) -> list:
    """
    Retrieve the names of all milestones from the JSON data.

    Args:
        data (dict): The JSON data.

    Returns:
        list: A list of milestone names.
    """
    return [milestone.get('name') for milestone in data.get('milestones', [])]


def get_milestone_by_name(data: dict, name: str) -> dict:
    """
    Retrieve a milestone from the JSON data by name.

    Args:
        data (dict): The JSON data.
        name (str): The name of the milestone to retrieve.

    Returns:
        dict: The milestone with the given name, or None if not found.
    """
    for milestone in data.get('milestones', []):
        if milestone.get('name') == name:
            return milestone
    return None

def get_milestone_details(data: dict, name: str, step_numbers: list) -> dict:
    """
    Return milestone object with specified steps.

    Args:
        data (dict): The JSON data.
        name (str): The name of the milestone.
        steps (list): List of step numbers to include.

    Returns:
        dict: The milestone details including name, description, and specified steps.
    """
    milestone = get_milestone_by_name(data, name)
    if milestone is None:
        return None

    selected_steps = [milestone['steps'][i-1] for i in step_numbers if 0 < i <= len(milestone['steps'])]
    return {
        'name': milestone['name'],
        'description': milestone['description'],
        'steps': selected_steps
    }
#%% User classifications
class UserIntent(dspy.Signature):
    """Clasify user message intent in response to context. What is the intent?"""
    
    context = dspy.InputField(desc="Context preceeding the user message.")
    message = dspy.InputField(prefix="User Message:")
    intent = dspy.OutputField(desc="Inquiry or Feedback or Direction or Confirmation or Other")
    
class UserComplexity(dspy.Signature):
    """Classify the complexity of the user message."""
    message = dspy.InputField(prefix="User Message:")
    complexity = dspy.OutputField(desc="Simple or Intermediate or Advanced")

class UserSentiment(dspy.Signature):
    """Analyze the sentiment of the user message as a response to the context. What is the sentiment?"""
    context = dspy.InputField(desc="Context preceding the user message.")
    message = dspy.InputField(prefix="User Message:")
    sentiment = dspy.OutputField(desc="Positive or Neutral or Negative or Mixed")

class UserEngagement(dspy.Signature):
    """Classify the user's engagement level."""
    context = dspy.InputField(desc="Context preceding the user message.")
    message = dspy.InputField(prefix="User Message:")
    engagement = dspy.OutputField(desc="Engaged or Participative or Passive or Disengaged")

# Assistant classifications

class CognitiveLoad(dspy.Signature):
    """Quantify the cognitive load of the text."""
    message = dspy.InputField(prefix="Assessed Text:")
    cognitive_load = dspy.OutputField(desc="Low, Medium, High")
    
    
class MilestoneName(dspy.Signature):
    """Identify the the milestone associated with the text prompt. What is the milestone name if any?"""
    message = dspy.InputField(prefix="Assessed Text:")
    milestone_array = dspy.InputField(prefix="milestones:",)
    milestone_name = dspy.OutputField(desc="milestone name or 'None'")
    
class MilestoneSteps(dspy.Signature):
    """Identify the steps cued by the text. List the step numbers."""
    message = dspy.InputField(prefix="Assessed Text:")
    steps = dspy.InputField()
    step_numbers = dspy.OutputField(desc="comma separated list of numbers")

class MilestoneRelevance(dspy.Signature):
    """Assess the relevance of the text to the milestone and steps. How relevant is the bulk of the text?"""
    message = dspy.InputField(prefix="Assessed Text:")
    milestone = dspy.InputField()
    relevance = dspy.OutputField(desc="Focused or Minor Deviation or Major Deviation")


# use when no milestone, or deviation detected
# if yes then user initiated deviation is true
# if no then irrelivant response
class ResponseRelevance(dspy.Signature):
    """Assess the relevance of the Text as a response to the Prompt. Is the text relevant to the prompt?"""
    context = dspy.InputField(prefix="Prompt:")
    message = dspy.InputField(prefix="Assessed Text:",desc="Response to the Prompt")
    relevance = dspy.OutputField(desc="Yes or No")

# use when user initiated deviation is true
#if yes then user attempted prompt hack
#if no then user irrelivant message
class IsPromptHackAttempt(dspy.Signature):
    """Inspect the text for signs of a prompt hack attempt. Is the text attempting to exploit LLM vulnerabilities?"""
    message = dspy.InputField(prefix="Assessed Text:")
    llm_exploit_attempted = dspy.OutputField(desc="Yes or No")
    

class AssessInitIntent(dspy.Signature):
    """Assess the message for intent. What is the intent?"""
    
    message = dspy.InputField()
    intent = dspy.OutputField(desc="Inquiry or Feedback or Direction or Confirmation or Other")
    
class AssessInitSatement(dspy.Signature):
    """Assess the message complexity. Is the message a short confirmation statement such as "begin", "start", "hi"?"""
    
    message = dspy.InputField()
    is_statement = dspy.OutputField(desc="yes or no")
    
    
    
class InitModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.statement = dspy.Predict(AssessInitSatement)
        self.intent = dspy.ChainOfThought(AssessInitIntent)
        self.prompthack = dspy.ChainOfThought(IsPromptHackAttempt)
        self.milestone_name_vote = Ensemble(reduce_fn=dspy.majority).compile([dspy.Predict(MilestoneName, n=3)])
    
    def forward(self, message):
        
        statement=self.statement(message=message).is_statement
        dspy.Suggest(
            statement.lower() in ['yes', 'no'],
            f"is_statement must be 'yes' or 'no'",
        )
        statement = statement.lower() == 'yes'
        intent = None
        milestone_name = None
        exploit_attempted = None
        if not statement:
            intent=self.intent(message=message).intent
            milestone_name = self.milestone_name_vote(message=message, milestone_array=f"{objective_object['milestones']}").milestone_name
            exploit_attempted = self.prompthack(message=message).llm_exploit_attempted.lower() == "yes"


        return dspy.Prediction(
            init=statement,
            intent=intent,
            milestone_relevance=milestone_name,
            prompt_hack_attempt=exploit_attempted
        )

def extract_number_prefix(step):
    """
    Extract the numeric part of a step.
    
    Parameters:
    step (str): The step string.
    
    Returns:
    int: The extracted step number.
    """
    match = re.match(r"(\d+)", step)
    return int(match.group(1)) if match else None

def compare_steps(current_steps, prev_steps):
    """
    Compare the steps of two milestones.
    
    Parameters:
    current_steps (list): The steps of the current milestone.
    prev_steps (list): The steps of the previous milestone.
    
    Returns:
    str: The progression status ("Progressive", "Regressive", or "Stationary").
    """
    current_step_numbers = [extract_number_prefix(step) for step in current_steps]
    prev_step_numbers = [extract_number_prefix(step) for step in prev_steps]
    
    max_current_step = max(current_step_numbers, default=0)
    max_prev_step = max(prev_step_numbers, default=0)
    
    if max_current_step > max_prev_step:
        return "Progressive"
    elif max_current_step < max_prev_step:
        return "Regressive"
    
    return "Stationary"

def get_array_position(data, name):
    for index, item in enumerate(data):
        if item['name'] == name:
            return index
    return -1  # Return -1 if the name is not found


def get_progression_status(milestone_arr, current_milestone_details, prev_milestone_details):
    """
    Get the progression status between two milestones.
    
    Parameters:
    current_milestone_details (dict): The details of the current milestone.
    prev_milestone_details (dict): The details of the previous milestone.
    
    Returns:
    str: The progression status ("Progressive", "Regressive", or "Stationary").
    """
    if not prev_milestone_details and current_milestone_details:
        return "Progressive"
    elif not current_milestone_details and prev_milestone_details:
        return "Regressive"
    elif not current_milestone_details and not prev_milestone_details:
        return "Stationary"
    
    # Extract milestone numbers
    current_number = get_array_position(milestone_arr, current_milestone_details['name'])
    prev_number = get_array_position(milestone_arr, prev_milestone_details['name'])
    
    if current_number > prev_number:
        return "Progressive"
    elif current_number < prev_number:
        return "Regressive"
    
    # If milestone numbers are the same, compare steps
    current_steps = current_milestone_details['steps']
    prev_steps = prev_milestone_details['steps']
    
    return compare_steps(current_steps, prev_steps)


class UserModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.intent = dspy.ChainOfThought(UserIntent)
        self.complexity = dspy.ChainOfThought(UserComplexity)
        self.sentiment = dspy.ChainOfThought(UserSentiment)
        self.engagement = dspy.ChainOfThought(UserEngagement)
    
    def forward(self, prev_assistant_message, message):
        context = prev_assistant_message
        # return {
        #     "intent": self.intent(context=context, message=message).intent,
        #     "complexity": self.complexity(message=message).complexity,
        #     "sentiment": self.sentiment(context=context, message=message).sentiment,
        #     "engagement": self.engagement(context=context, message=message).engagement
        # }
        return dspy.Prediction(
            intent=self.intent(context=context, message=message).intent,
            complexity=self.complexity(message=message).complexity,
            sentiment=self.sentiment(context=context, message=message).sentiment,
            engagement=self.engagement(context=context, message=message).engagement
        )
        # TODO why is engagement negative? eg. context forgotten or ignored.

with open(objective_path, 'r', encoding='utf-8') as file:
    objective_object = json.load(file)

class AssistantModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.milestone_name = dspy.ChainOfThought(MilestoneName)
        self.milestone_name_vote = Ensemble(reduce_fn=dspy.majority).compile([dspy.Predict(MilestoneName, n=3)])
        self.milestone_steps = dspy.ChainOfThought(MilestoneSteps)
        self.milestone_relevance = dspy.ChainOfThought(MilestoneRelevance)
        self.cognitive_load = dspy.ChainOfThought(CognitiveLoad)
        self.response_relevance = dspy.ChainOfThought(ResponseRelevance)
        self.prompt_hack = dspy.ChainOfThought(IsPromptHackAttempt)
    
    def forward(self, prev_user_message, message, prev_milestone_details = False):
        context = prev_user_message
        cognitive_load = self.cognitive_load(message=message).cognitive_load
        milestone_names = get_milestone_names(objective_object)
        milestone_name = self.milestone_name_vote(message=message, milestone_array=f"{objective_object['milestones']}").milestone_name
        if milestone_name not in milestone_names or milestone_name == "None":
            # print("Milestone not detected: ", milestone_name)
            with dspy.context(lm=gpt4o):
                milestone_name = self.milestone_name(message=message, milestone_array=f"{objective_object['milestones']}").milestone_name
            # print("Milestone from 4o", milestone_name)
        dspy.Suggest(
            milestone_name in milestone_names or milestone_name == "None",
            f"milestone_name must be one of {milestone_names} or None",
        )
        # print(milestone_name)
        milestone_detected = milestone_name in milestone_names
        milestone_details = None
        milestone_relevance = None
        deviation_trigger = None
        if milestone_detected:
            milestone = get_milestone_by_name(objective_object, milestone_name)
            milestone_str = json.dumps(milestone)
            milestone_steps = self.milestone_steps(message=message, steps=milestone_str).step_numbers
            milestone_steps = milestone_steps.replace(" ", "").replace("[", "").replace("]", ""). replace(".", "")
            if not milestone_steps.replace(',', '').isdigit():
                print("fail parse steps", milestone_steps)
            dspy.Suggest(
                milestone_steps.replace(',', '').isdigit(),
                f"milestone_steps must be list of numbers separated by commas",
            )
            milestone_steps_arr = [int(i) for i in milestone_steps.split(",")]

            # should assert steps are in the range of the milestone steps
            milestone_details = get_milestone_details(objective_object, milestone_name, milestone_steps_arr)
            milstone_details_str = json.dumps(milestone_details)
            milestone_relevance = self.milestone_relevance(message=message, milestone=milstone_details_str).relevance
            
        if not milestone_detected or milestone_relevance.lower() == "major deviation":
            response_relevance = self.response_relevance(context=context, message=message).relevance.lower() == "yes"
            exploit_attempted = False
            if response_relevance:
                exploit_attempted = self.prompt_hack(message=context).llm_exploit_attempted.lower() == "yes"
            deviation_trigger = "Prompt Hack Attempt" if exploit_attempted else "Irrelevant Response" if not response_relevance else "User Tangent"
                
        progresion_status = get_progression_status(objective_object['milestones'], milestone_details, prev_milestone_details)

        # assessment_obj={
        #     "milestone_details": milestone_details,
        #     "cognitive_load": cognitive_load,
        #     "milestone_relevance": milestone_relevance,
        #     "deviation_trigger": deviation_trigger,
        #     "progression_status": progresion_status
        # }
        return dspy.Prediction(milestone_details=milestone_details, cognitive_load=cognitive_load, milestone_relevance=milestone_relevance, deviation_trigger=deviation_trigger, progression_status=progresion_status)
       

assess_assistant_msg = assert_transform_module(AssistantModule(), backtrack_handler)
assess_user_msg = assert_transform_module(UserModule(), backtrack_handler)
assess_init_statement = assert_transform_module(InitModule(), backtrack_handler)

# %%

def assess_thread(test_thread: ThreadObject, assistant_program=assess_assistant_msg, user_program=assess_user_msg, init_program=assess_init_statement):
    print("assessing thread: ",test_thread.thread_id)
    prev_milestone_details = False
    test_thread.messages[0]["assessment"]=init_program(test_thread.messages[0]["message"]).toDict()
    cog_load_arr =[]
    sentiment_arr=[]
    engagement_arr=[]
    highest_milestone_details = None
    for i in range(1, len(test_thread.messages)):
        context = test_thread.messages[i-1]["message"]
        current_msg = test_thread.messages[i]["message"]
        assistant_current = current_msg if test_thread.messages[i]["role"] == "assistant" else False
        if assistant_current:
            test_thread.messages[i]["assessment"] = assistant_program(prev_user_message = context, message = assistant_current, prev_milestone_details=prev_milestone_details).toDict()
            prev_milestone_details = test_thread.messages[i]["assessment"]["milestone_details"]
            cog_load_arr.append(test_thread.messages[i]["assessment"]["cognitive_load"])
            if get_progression_status(objective_object['milestones'], test_thread.messages[i]["assessment"]["milestone_details"], highest_milestone_details) == "Progressive":
                highest_milestone_details = test_thread.messages[i]["assessment"]["milestone_details"]
        else:
            test_thread.messages[i]["assessment"] = user_program(prev_assistant_message=context, message= current_msg).toDict()
            sentiment_arr.append(test_thread.messages[i]["assessment"]["sentiment"])
            engagement_arr.append(test_thread.messages[i]["assessment"]["engagement"])
            
            
    highest_step = None
    if highest_milestone_details:
        highest_step = {
            "milestone": highest_milestone_details["name"],
            "step": highest_milestone_details["steps"][-1]
        }         
    if cog_load_arr:
        test_thread.cog_load_majority = max(set(cog_load_arr), key=cog_load_arr.count)
    else:
        test_thread.cog_load_majority = None

    if sentiment_arr:
        test_thread.sentiment_majority = max(set(sentiment_arr), key=sentiment_arr.count)
    else:
        test_thread.sentiment_majority = None

    if engagement_arr:
        test_thread.engagement_majority = max(set(engagement_arr), key=engagement_arr.count)
    else:
        test_thread.engagement_majority = None
    test_thread.highest_step = highest_step
    return test_thread
# %%
def assess_thread_by_id(thread_id):
    test_thread = next(thread for thread in threads if thread.thread_id == thread_id)
    return assess_thread(test_thread)


def get_most_common_milestone_and_step(highest_milestone_details_arr):
    # Count the occurrences of each milestone
    milestone_counter = Counter(item['milestone'] for item in highest_milestone_details_arr)

    # Get the most common milestone
    most_common_milestone = milestone_counter.most_common(1)[0][0]

    # Filter the array to only include items with the most common milestone
    most_common_milestone_items = [item for item in highest_milestone_details_arr if item['milestone'] == most_common_milestone]

    # Count the occurrences of each step within the most common milestone
    step_counter = Counter(item['step'] for item in most_common_milestone_items)

    # Get the most common step
    most_common_step = step_counter.most_common(1)[0][0]

    return {"milestone":most_common_milestone, "step":most_common_step}


def get_full_assessment(threads, assistant_program=assess_assistant_msg, user_program=assess_user_msg, init_program=assess_init_statement):
    total = {}
    total["thread_count"] = len(threads)
    total["engagement_duration"] = 0
    total["message_count"] = 0
    total["tokens"]=0
    total["highest_thread_tokens"]=0
    total["leads"]=0

    cog_load_arr =[]
    sentiment_arr=[]
    engagement_arr=[]
    highest_milestone_details_arr = []
    for thread in threads:
        thread = assess_thread(thread, assistant_program, user_program, init_program)
        
        thread.message_count=len(thread.messages)
        total["message_count"] += thread.message_count
        finished_time = datetime.strptime(thread.finished_time, '%Y%m%d%H%M%S')
        started_time = datetime.strptime(thread.started_time, '%Y%m%d%H%M%S')

        # Subtract to get the duration in seconds
        thread.engagement_duration = ( finished_time - started_time).total_seconds()

        total["engagement_duration"] += thread.engagement_duration
        thread.tokens=0
        
        cog_load_arr.append(thread.cog_load_majority)
        sentiment_arr.append(thread.sentiment_majority)
        engagement_arr.append(thread.engagement_majority)
        highest_milestone_details_arr.append(thread.highest_step)
        
        thread.email=None
        
        for message in thread.messages:
            # does message contain email address
            thread.tokens+=message["tokens"]
            total["tokens"]+=message["tokens"]
            email_matches = re.findall(email_pattern, message["message"])
            if email_matches:
                for email in email_matches:
                    print(email)
                    thread.email=email
                    
        if thread.tokens > total["highest_thread_tokens"]:
            total["highest_thread_tokens"] = thread.tokens
        if thread.email:
            total["leads"]+=1
                    
    aggregate = {}
    aggregate["message_count"] = total["message_count"]/total["thread_count"]
    aggregate["engagement_duration"] = total["engagement_duration"]/total["thread_count"]
    aggregate["tokens"] = total["tokens"]/total["thread_count"]         
    aggregate["cognitive_load"] = max(set(cog_load_arr), key=cog_load_arr.count)
    aggregate["sentiment"] = max(set(sentiment_arr), key=sentiment_arr.count)
    aggregate["engagement"] = max(set(engagement_arr), key=engagement_arr.count)
    
    # Filter out None items
    filtered_highest_milestone_details_arr = [item for item in highest_milestone_details_arr if item is not None]

    aggregate["dropoff_point"]=get_most_common_milestone_and_step(filtered_highest_milestone_details_arr)
    
    # TODO get need to get in and out token count to calculate true pricing
    cost_multiplier = 80000
    conversion_rate = round((total['leads']/total['thread_count']) * 100, 2)

    print(f"total engagements: {round(total['thread_count'])}")
    print(f"total engagement duration: {round(total['engagement_duration']/60)} minutes")
    print(f"average engagement duration: {round(aggregate['engagement_duration']/60)} minutes\n")
    print(f"total convert to lead: {round(total['leads'])}")
    print(f"conversion rate: {conversion_rate}%\n")
    print(f"total spend: ${round(total['tokens']/cost_multiplier, 2)}")
    print(f"average spend per engagement: ${round(aggregate['tokens']/cost_multiplier, 2)}")
    print(f"highest engagement spend: ${round(total['highest_thread_tokens']/cost_multiplier, 2)}\n")
    print(f"total message count: {round(total['message_count'])}")
    print(f"average message count: {round(aggregate['message_count'])}\n")
    print(f"average cognitive load: {aggregate['cognitive_load']}")
    print(f"average sentiment: {aggregate['sentiment']}")
    print(f"average engagement: {aggregate['engagement']}")
    print(f"most common dropoff point: {aggregate['dropoff_point']}")
    

    return total, aggregate, threads
# %%


def gen_trainset(message_arr_arr):
    user_trainset = []
    assistant_trainset = []

    for test_thread in message_arr_arr:
        prev_milestone_details = False
        for i in range(1, len(test_thread)):
            context = test_thread[i-1]["message"]
            current_msg = test_thread[i]["message"]
            assistant_current = current_msg if test_thread[i]["role"] == "assistant" else False
            assessment_obj = test_thread[i]["assessment"]
            if assistant_current:
                assistant_trainset.append(dspy.Example({"prev_user_message": context,
                                                        "message": assistant_current,
                                                        "prev_milestone_details": prev_milestone_details,
                                                        "prev_user_message": context,
                                                        "milestone_details": assessment_obj["milestone_details"],
                                                        "cognitive_load": assessment_obj["cognitive_load"],
                                                        "milestone_relevance": assessment_obj["milestone_relevance"],
                                                        "deviation_trigger": assessment_obj["deviation_trigger"],
                                                        "progression_status": assessment_obj["progression_status"]
                                                        }).with_inputs('prev_user_message', 'message', 'prev_milestone_details'))                                                                            
                prev_milestone_details = assessment_obj["milestone_details"]
            else:
                user_trainset.append(dspy.Example({"prev_assistant_message": context,
                                                    "message": current_msg,
                                                    "intent": assessment_obj["intent"],
                                                    "complexity": assessment_obj["complexity"],
                                                    "sentiment": assessment_obj["sentiment"],
                                                    "engagement": assessment_obj["engagement"]
                                                    }).with_inputs('prev_assistant_message', 'message'))
    return user_trainset, assistant_trainset        
                

def validate_user_assess(example, pred, trace=None):
    intent = example.intent == pred.intent
    complexity = example.complexity == pred.complexity
    sentiment = example.sentiment == pred.sentiment
    engagement = example.engagement == pred.engagement
    
    if trace is None:
        return (intent + complexity + sentiment + engagement) / 4.0
    else:
        return intent and complexity and sentiment and engagement
    
def validate_assistant_assess(example, pred, trace=None):
    milestone_details = example.milestone_details == pred.milestone_details
    cognitive_load = example.cognitive_load == pred.cognitive_load
    milestone_relevance = example.milestone_relevance == pred.milestone_relevance
    deviation_trigger = example.deviation_trigger == pred.deviation_trigger
    progression_status = example.progression_status == pred.progression_status
    
    if trace is None:
        return (milestone_details + cognitive_load + milestone_relevance + deviation_trigger + progression_status) / 5.0
    else:
        return milestone_details and cognitive_load and milestone_relevance and deviation_trigger and progression_status


def evaluate_model(program, trainset, metric):
    # Set up the evaluator, which can be re-used in your code.
    evaluate = Evaluate(devset=trainset, num_threads=2, display_progress=True, display_table=5)
    # Launch evaluation.
    return evaluate(program, metric)
    
def optimise_program_fewshot(program, trainset, metric):
    fewshot_optimizer = BootstrapFewShot(metric, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=5, teacher_settings=dict(lm=gpt4o))
    program_compiled = fewshot_optimizer.compile(student = program, trainset=trainset)
    return program_compiled

def optimise_program_fewshot_rnd(program, trainset, metric):
    # Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
    # The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.
    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=4)
    teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, **config)
    optimized_program = teleprompter.compile(program, trainset=trainset)
    return optimized_program


# %% 
# TODO add scotts response and others to the trainset
user_trainset, assistant_trainset = gen_trainset(labeled_threads)

def run_optimise():
    optimised_assess_user = optimise_program_fewshot(assess_user_msg, user_trainset, validate_user_assess)
    optimised_assess_assistant = optimise_program_fewshot(assess_assistant_msg, assistant_trainset, validate_assistant_assess)
    optimised_assess_user_rnd = optimise_program_fewshot_rnd(assess_user_msg, user_trainset, validate_user_assess)
    optimised_assess_assistant_rnd = optimise_program_fewshot_rnd(assess_assistant_msg, assistant_trainset, validate_assistant_assess)

    eval_user = evaluate_model(assess_user_msg, user_trainset, validate_user_assess)  
    eval_assistant = evaluate_model(assess_assistant_msg, assistant_trainset, validate_assistant_assess)
    eval_user_optimised = evaluate_model(optimised_assess_user, user_trainset, validate_user_assess)  
    eval_assistant_optimised = evaluate_model(optimised_assess_assistant, assistant_trainset, validate_assistant_assess)
    eval_user_optimised_rnd = evaluate_model(optimised_assess_user_rnd, user_trainset, validate_user_assess)
    eval_assistant_optimised_rnd = evaluate_model(optimised_assess_assistant_rnd, assistant_trainset, validate_assistant_assess)

    print("eval_user: ", eval_user)
    print("eval_user_optimised: ", eval_user_optimised)
    print("eval_user_opt_rnd: ", eval_user_optimised_rnd)
    print("")
    print("eval_assistant: ", eval_assistant)
    print("eval_assistant_optimised: ", eval_assistant_optimised)
    print("eval_assistant_opt_rnd", eval_assistant_optimised_rnd)

    # eval_user:  81.08
    # eval_user_optimised:  81.76
    # eval_user_opt_rnd:  81.08

    # eval_assistant:  73.95
    # eval_assistant_optimised:  85.12
    # eval_assistant_opt_rnd: 86.05

    optimised_assess_user.save("programs/optimised_assess_user.json")
    optimised_assess_assistant.save("programs/optimised_assess_assistant.json")
    optimised_assess_user_rnd.save("programs/optimised_assess_user_rnd.json")
    optimised_assess_assistant_rnd.save("programs/optimised_assess_assistant_rnd.json")

# %%
user_program = assess_user_msg
user_program.load("programs/optimised_assess_user.json")

assistant_program = assess_assistant_msg
assistant_program.load("programs/optimised_assess_assistant_rnd.json")

# %%
total, aggregate, threads_data = get_full_assessment(threads, assistant_program, user_program, assess_init_statement)
# To save the object to a JSON file
with open('threads_data.json', 'w') as file:
    json.dump({"total":total,"aggregate":aggregate,"threads":[t.to_dict() for t in threads_data]}, file, indent=4, ensure_ascii=False)


# %%
