import random

def create_daily_meal_plan(matching_recipes):
    """
    Given a list of matching recipes (strings),
    organize them into breakfast, lunch, and dinner.
    """
    meal_plan = {}

    # Shuffle recipes to randomize
    random.shuffle(matching_recipes)

    # Assign meals (simple rule: first 3 recipes)
    if len(matching_recipes) >= 3:
        meal_plan["Breakfast"] = matching_recipes[0]
        meal_plan["Lunch"] = matching_recipes[1]
        meal_plan["Dinner"] = matching_recipes[2]
    else:
        # If fewer than 3 recipes, reuse some
        meal_plan["Breakfast"] = matching_recipes[0] if len(matching_recipes) > 0 else "No recipe found."
        meal_plan["Lunch"] = matching_recipes[1] if len(matching_recipes) > 1 else matching_recipes[0]
        meal_plan["Dinner"] = matching_recipes[2] if len(matching_recipes) > 2 else matching_recipes[0]

    return meal_plan