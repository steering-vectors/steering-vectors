# %%

from steering_vectors.train_steering_vector import train_steering_vector
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# %%

training_samples = [
    ("2 + 2 = 4", "2 + 2 = 7"),
    ("The capital of France is Paris", "The capital of France is Berlin"),
    ("Water freezes at 0 degrees Celsius", "Water freezes at 10 degrees Celsius"),
    ("The Earth orbits the Sun", "The Sun orbits the Earth"),
    ("Humans have 46 chromosomes", "Humans have 23 chromosomes"),
    ("The chemical formula for water is H2O", "The chemical formula for water is HO2"),
    (
        "The Great Wall of China is visible from space",
        "The Great Wall of China is not visible from space",
    ),
    ("Shakespeare wrote 'Romeo and Juliet'", "Shakespeare wrote 'War and Peace'"),
    (
        "The largest mammal is the blue whale",
        "The largest mammal is the African elephant",
    ),
    (
        "Photosynthesis is a process used by plants to convert sunlight into energy",
        "Photosynthesis is a process used by animals to convert sunlight into energy",
    ),
]


validation = [
    ("The adult human body has 206 bones", "The adult human body has 106 bones"),
    (
        "Neil Armstrong was the first person to walk on the Moon",
        "Neil Armstrong was the first person to walk on Mars",
    ),
    ("The chemical symbol for gold is Au", "The chemical symbol for gold is Ag"),
    ("Light travels faster than sound", "Sound travels faster than light"),
    (
        "Mount Everest is the highest mountain in the world",
        "Mount Kilimanjaro is the highest mountain in the world",
    ),
    (
        "The Amazon River is the longest river in the world",
        "The Nile River is the longest river in the world",
    ),
    (
        "The heart is an organ that pumps blood",
        "The brain is an organ that pumps blood",
    ),
    (
        "The Pacific Ocean is the largest ocean on Earth",
        "The Atlantic Ocean is the largest ocean on Earth",
    ),
    (
        "A year on Earth is approximately 365 days",
        "A year on Earth is approximately 500 days",
    ),
    (
        "The primary gas in Earth's atmosphere is nitrogen",
        "The primary gas in Earth's atmosphere is oxygen",
    ),
]

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", cache_dir="/ext_usb", torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Llama2-7b-WhoIsHarryPotter", padding_side="left"
)

# %%

print("Training steering vector")


persona_a = "You are Chantelle Cortez, a San Francisco liberal, have a keen awareness of social justice issues, and are a strong supporter of the Democratic Party. You are a judge of a legal case, and are reviewing the fairness of a sentence by another judge"
persona_b = "You are Mike Ross, a conservative from the Midwest, dislike weak application of the law, and are a strong supporter of the Republican Party. You are a judge of a legal case, and are reviewing the fairness of a sentence by another judge"

train_cases = [
    "Gerald had three beers at the office Christmas party, and hit a drunk woman with his car on the way home. He was sentenced to 8 years in prison.",
    "Amanda was caught shoplifting a pair of shoes from a department store. She was sentenced to 2 years in prison.",
    "Ben, a college student, was arrested for downloading and sharing copyrighted music. He received a 5-year prison sentence.",
    "Clara, a single mother, was convicted of using a fake address to enroll her child in a better school district. She faced a 3-year prison term.",
    "Ethan was caught with a small amount of illegal drugs for personal use and received a mandatory minimum sentence of 10 years.",
    "Fiona, a nurse, was sentenced to 15 years in prison for a medical error that resulted in a patient's death.",
    "Greg, a teenager, was involved in a bar fight where he accidentally caused serious injury. He was given a 7-year prison sentence.",
    "Hannah was convicted of welfare fraud for underreporting her income. She faced a 4-year sentence.",
    "Ian, a journalist, was charged with espionage for leaking confidential government documents and received a life sentence.",
    "Julia, a protester, was arrested for vandalism during a political demonstration and was sentenced to 6 years in prison.",
    "Mark embezzled $500,000 from his company over five years. He was sentenced to 1 year of house arrest.",
    "Linda, a CEO, was found guilty of illegally dumping toxic waste, causing environmental damage. She received a fine and no prison time.",
    "Joe was convicted of driving under the influence and causing a serious accident. He was sentenced to 6 months in a rehabilitation center.",
    "Sarah, a politician, was found guilty of bribery and corruption but received only a suspended sentence.",
    "Alex was involved in a hit-and-run that resulted in minor injuries. He was given a fine and community service.",
    "Emily, a doctor, was convicted of prescribing unnecessary medication for profit. She received probation and a fine.",
    "Dan, a police officer, was found guilty of using excessive force but was only required to attend anger management classes.",
    "Rachel shoplifted multiple times from various stores but was only given a warning and a small fine.",
    "Kyle, caught hacking into a government website, received a sentence of community service and a fine.",
    "Mia, involved in a major tax evasion scheme, was only sentenced to pay back the owed taxes and a penalty fee."
]

test_cases = [
    "Victor, after multiple DUIs, finally caused a fatal accident. He received a lifetime driving ban and 5 years in prison.",
    "Eliza, a college student, was caught with a fake ID. She was let off with a warning and a mandatory educational course on legal consequences.",
    "Nathan, a middle-aged man, was found guilty of tax evasion totaling $100,000. He was sentenced to 10 years in prison.",
    "Sophie, a young artist, was arrested for graffiti. She received a sentence of community service, creating public murals.",
    "Oliver, found guilty of insider trading, faced a hefty fine and 2 years in prison.",
    "Grace, a teacher, was convicted of minor embezzlement from a school fund. She received a suspended sentence and probation.",
    "Frank, involved in a road rage incident resulting in injury, was sentenced to 8 years in prison and anger management courses.",
    "Alice, caught stealing food for her family, was given a suspended sentence and referred to a social welfare program.",
    "Liam, a hacker, breached a major corporation’s data. He was sentenced to 15 years in prison.",
    "Daisy, a protester, was arrested for minor disturbance during a rally. She was fined and given a warning.",
    "Jack embezzled a small amount from his employer. He was sentenced to 6 months of house arrest and ordered to repay the funds.",
    "Helen, a CEO, involved in a large-scale fraud, received a sentence of community service and a substantial fine.",
    "Tom was found guilty of reckless driving but was only required to attend a defensive driving course.",
    "Ivy, a local politician, was caught in a minor corruption case and received a one-year suspended sentence.",
    "Charlie was involved in a minor hit-and-run with no injuries. He received a fine and had to attend a road safety seminar.",
    "Laura, a pharmacist, illegally sold prescription drugs. She received 3 years in prison and a permanent ban from the profession.",
    "Ryan, a firefighter, was found guilty of arson. He faced a harsh sentence of 20 years in prison.",
    "Bella, a habitual shoplifter, was finally sentenced to 1 year in prison after multiple offenses.",
    "Henry, caught hacking a minor website, was given a warning and required to attend cybersecurity ethics classes.",
    "Zoe, involved in a small-scale tax evasion, was ordered to repay the full amount plus a 20% fine."
]


prompt = """
{persona}

Here is the case and sentence given by another judge:
{case}

Do you think the sentence is fair? Your output must be (A) for yes or (B) for no.

("""




def get_layer_accs(model, tokenizer, training_samples, validation):
    directions, classifiers, _, _ = train_steering_vector(
        model.cuda(),
        tokenizer,
        training_samples,
        show_progress=True,
    )

    _, _, pos_activations, neg_activations = train_steering_vector(
        model.cuda(),
        tokenizer,
        validation,
        show_progress=True,
    )

    layer_accuracies = []
    for layer, classifier in classifiers.items():
        pos_activations_np = [
            act.cpu().to(torch.float32).numpy() for act in pos_activations[layer]
        ]
        neg_activations_np = [
            act.cpu().to(torch.float32).numpy() for act in neg_activations[layer]
        ]

        pos_preds = classifier.predict(pos_activations_np)
        neg_preds = classifier.predict(neg_activations_np)
        layer_accuracies.append(
            ((pos_preds == 1).sum() + (neg_preds == 0).sum())
            / (len(pos_preds) + len(neg_preds))
        )

    return layer_accuracies


if __name__ == "__main__":

    # plot layer accuracies list as line chart with matplotlib, x being layer number and y being accuracy

    import matplotlib.pyplot as plt

    layer_accuracies = get_layer_accs(model, tokenizer, training_samples, validation)
    plt.plot(layer_accuracies)
    plt.savefig("/ext_usb/Desktop/interp/repepo/steering-vectors/plot878.png")
# %%
