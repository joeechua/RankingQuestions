import json
all_data = {}
from preprocess.descriptors import SentimentPreProcessor, StrategiesPreProcessor, TopicsPreProcessor, cosine_sim

sloganfile = open("data/annotations/Slogans.json")
slogans = json.load(sloganfile)
sentimentfile = open("data/annotations/Sentiments.json")
sentiments = json.load(sentimentfile)
strategiesfile = open("data/annotations/Strategies.json")
strategies = json.load(strategiesfile)
topicsfile = open("data/annotations/Topics.json")
topics = json.load(topicsfile)
qafile = open("data/annotations/QA_Combined_Action_Reason.json")
qas = json.load(qafile)

def transform(target_lst, descriptor):
    """
    Transform the target_lst of topics provided by the PITTs dataset to
    a Pytorch tensor based on the sBERT model.

    target_list: a list of lists, each element may contain a number or text
    """

    # flatten list
    target_lst = [item for sublist in target_lst for item in sublist]

    count = 0
    vec_lst = []
    num_lst = []

    if descriptor == "sentiment":
        proc = SentimentPreProcessor()
    elif descriptor == "topic":
        proc = TopicsPreProcessor()
    else:
        proc = StrategiesPreProcessor()

    for el in target_lst:
        try:
            x = int(el)
            num_lst.append(x)
            count += 1
        except ValueError:
            # Get the vector representation of this phrase
            vec_lst.append(proc.text_embed_model.get_vector_rep(el))

    if count == 0:
        # The target list has all user text inputs so try to find the
        # most represented phrase
        cosines = [0] * len(vec_lst)
        for i in range(len(vec_lst)):
            for j in range(len(vec_lst)):
                if i != j:
                    cosines[i] += cosine_sim(vec_lst[i], vec_lst[j])

        max_val = max(cosines)
        max_index = cosines.index(max_val)

        final = target_lst[max_index]

    else:
        if 0 in num_lst:
            num_lst.remove(0)
        final = proc.id_to_word[max(num_lst, key=num_lst.count)]

    return final

main_idx = 1
for k in list(slogans.keys()):
    try:
        sentiment = sentiments[k]
        strategy = strategies[k]
        topic = topics[k]
        qa = qas[k]
        slogan = slogans[k]

        #get top id for sentiments, strategies, topics
        sent = transform(sentiment, "sentiment")
        strat = transform(strategy, "strategy")
        top = transform(topic, "topic")
        #combine best s,s,t with each combination of qa and slogan
        for question in qa:
            for slogan in slogans:
                all_data[main_idx] = {"Slogan id": main_idx, "Image": k, "Sentiment": sentiment, "Strategy": strategy, "Topic":topic, "QA": question, "Slogan": slogan}
                main_idx += 1

    except KeyError:
        next


json_object = json.dumps(all_data)
with open("slogan_descriptor_combos.json", "w") as outfile:
    outfile.write(json_object)
outfile.close()