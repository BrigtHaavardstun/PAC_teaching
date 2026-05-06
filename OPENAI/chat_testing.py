from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def run(text):
    client = OpenAI()

    messages: list[ChatCompletionMessageParam] = [
        {

            "role": "user",
            "content": text,
        }
    ]

    response = client.chat.completions.create(
        model='gpt-5.4-nano',
        messages=messages,
        temperature=1,
        top_p=1,
        max_completion_tokens=600,
        reasoning_effort="none")

    text = response.choices[0].message.content
    return text


def run_sys(text, sys_text):
    client = OpenAI()

    messages: list[ChatCompletionMessageParam] = [
        {

            "role": "system",
            "content": sys_text,
        },
        {

            "role": "user",
            "content": text,
        }
    ]

    response = client.chat.completions.create(
        model='gpt-5.4-nano',
        messages=messages,
        temperature=1,
        top_p=1,
        max_completion_tokens=600,
        reasoning_effort="none")

    text = response.choices[0].message.content
    return text


def main():
    xs = [50, 22, 104, 130, 169, 221]
    C = [5, 11, 13]
    for x in xs:

        text = "Is {c} a divisor of {x}?"
        sys_text = "Answer format: 'Answer = Yes' or 'Answer = No'"

        sys_text2 = f"Answer format: {' or '.join([f'\'Answer = {c}\'' for c in C])}."
        text2 = f"Which of {', '.join(map(str, C[:-1]))} and {C[-1]} is a divisor of {x}?"

        print("#"*5, "ERROR ESTIMATE", "#"*5)
        print("System:", sys_text)

        for c in C:
            text_curr = text.format(x=x, c=c)
            print("Q:", text_curr)
            print("A:", run_sys(text=text_curr, sys_text=sys_text))

        print("Q:", text2, "System:", sys_text2)
        print(run_sys(text2, sys_text2))


if __name__ == "__main__":
    main()
