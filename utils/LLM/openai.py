from openai import OpenAI

def gpt_4o_mini(system_content, user_content):

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )

    return completion.choices[0].message.content