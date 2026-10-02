---
title: FoodExtract Fine-tuned LLM Structued Data Extractor v1
emoji: 📝➡️🍟
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

"""
Fine-tuned Gemma 3 270M to extract food and drink items from raw text.

Input can be any form of real text (mostly focused on shorter image caption-like texts):

```
A truly eclectic and mouth-watering feast is laid out on the table, featuring savory favorites like crispy fried chicken, 
a perfectly seared steak, and loaded tacos, complete with a side of creamy mayonnaise. To balance the heavier mains, 
a vibrant assortment of fresh fruit sits nearby, including a crisp red apple, a tropical pineapple, and a scattering of 
sweet cherries. Thirst-quenching options complete this extravagant spread, with a classic iced latte, an earthy matcha latte, 
and a simple, refreshing glass of milk ready to be enjoyed.
```

And output will be a formatted string such as the following:

```
food_or_drink: 1
tags: fi, re
foods: tacos,red apple, pineapple, cherries, fried chicken, steak, mayonnaise
drinks: iced latte, matcha latte, milk
```

The tags map to the following items:

```
tags_dict = {'np': 'nutrition_panel',
 'il': 'ingredient list',
 'me': 'menu',
 're': 'recipe',
 'fi': 'food_items',
 'di': 'drink_items',
 'fa': 'food_advertistment',
 'fp': 'food_packaging'}
```

* You can see walkthrough step by step code details at: https://www.learnhuggingface.com/notebooks/hugging_face_llm_full_fine_tune_tutorial 
* See the fine-tuning dataset: https://huggingface.co/datasets/mrdbourke/FoodExtract-1k
* See the fine-tuned model: https://huggingface.co/mrdbourke/FoodExtract-gemma-3-270m-fine-tune-v1
"""
