from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import torch
from dotenv import load_dotenv
import os
import json
import time
from collections import defaultdict
from googletrans import Translator
from langdetect import detect

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='images')
CORS(app, resources={r"/*": {"origins": "*"}})

# Load product data
def load_product_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "data.json")
        print(f"Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            print("data.json not found!")
            return {"products": [], "categories": {}, "training_examples": [], "support_info": {}}
            
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            print(f"Loaded {len(data['products'])} products")
            return data
    except Exception as e:
        print(f"Error loading product data: {str(e)}")
        return {"products": [], "categories": {}, "training_examples": [], "support_info": {}}


class ProductAI:
    def __init__(self):
        try:
            print("Initializing AI models...")
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load data and initialize
            self.product_data = load_product_data()
            self.conversation_history = defaultdict(list)
            self.user_preferences = defaultdict(dict)
            self.product_embeddings = self._generate_embeddings()
            
            print("AI models initialized successfully!")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Error initializing AI: {str(e)}")
            self.model_loaded = False

    def _generate_embeddings(self):
        """Generate embeddings for all products"""
        embeddings = {}
        for product in self.product_data["products"]:
            text = f"{product['name']} {product['description']} {' '.join(product.get('keywords', []))}"
            embedding = self.nlp_model.encode(text, convert_to_tensor=True)
            embeddings[product['id']] = embedding
        return embeddings

    def process_customer_query(self, query, session_id=None):
        """Amazon-style query processing with NLP and personalization"""
        try:
            # 1. Store query in conversation history for context
            if session_id:
                self.conversation_history[session_id].append({
                    'query': query,
                    'timestamp': time.time()
                })
            
            # 2. Understand Customer Query - NLP Intent Recognition
            intent = self.analyze_intent(query)
            print(f"Detected intent: {intent['type']} with confidence {intent['confidence']:.2f}")
            
            # 3. Process based on intent with personalization
            if intent['type'] == 'product_search':
                return self.handle_product_search(query, session_id)
            elif intent['type'] == 'product_comparison':
                return self.handle_product_comparison(query, session_id)
            elif intent['type'] == 'recommendation':
                return self.handle_recommendation(query, session_id)
            elif intent['type'] == 'order_tracking':
                return self.handle_order_tracking(query, session_id)
            elif intent['type'] == 'customer_service':
                return self.handle_customer_service(query)
            else:
                # 4. Generate conversational response for general queries
                return self.generate_conversational_response(query, session_id)
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return self.get_smart_fallback_response()

    def analyze_intent(self, query):
        """Advanced intent analysis with confidence scoring"""
        query_lower = query.lower()
        
        # Define intent patterns with keywords and phrases
        intents = {
            'product_search': ['show', 'find', 'looking for', 'search', 'browse'],
            'product_comparison': ['compare', 'difference', 'better', 'vs', 'versus'],
            'recommendation': ['recommend', 'suggest', 'best', 'top', 'popular'],
            'order_tracking': ['order', 'track', 'shipping', 'delivery', 'package'],
            'customer_service': ['help', 'support', 'issue', 'problem', 'question']
        }
        
        # Calculate intent scores
        intent_scores = {}
        for intent, keywords in intents.items():
            score = sum(3 if keyword in query_lower else 0 for keyword in keywords)
            # Add semantic similarity using embeddings
            query_embedding = self.nlp_model.encode(query_lower, convert_to_tensor=True)
            for keyword in keywords:
                keyword_embedding = self.nlp_model.encode(keyword, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(query_embedding, keyword_embedding).item()
                score += similarity * 2
            intent_scores[intent] = score
        
        # Get highest scoring intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence (normalize score)
        max_possible = 15  # Maximum possible score
        confidence = min(best_intent[1] / max_possible, 1.0)
        
        return {
            'type': best_intent[0] if confidence > 0.3 else 'general',
            'confidence': confidence,
            'all_scores': intent_scores
        }

    def handle_product_search(self, query, session_id):
        """Enhanced product search with personalization and follow-up questions"""
        # Find matching products
        matches = self.find_matching_products(query)
        
        # Apply personalization if we have user history
        if session_id and session_id in self.user_preferences:
            matches = self.personalize_results(matches, session_id)
        
        if matches:
            # Determine if we should ask a follow-up question
            should_ask_followup = len(matches) > 3
            
            return {
                "response": {
                    "type": "product_results",
                    "message": f"I found {len(matches)} products that match your search:",
                    "products": [self.format_product_details(p) for p in matches[:3]],
                    "total_matches": len(matches),
                    "follow_up_question": "Would you like to narrow down these results?" if should_ask_followup else None,
                    "filter_options": self.generate_filter_options(matches) if should_ask_followup else [],
                    "suggestions": [
                        "Show more details",
                        "Compare top results",
                        "Sort by price"
                    ]
                }
            }
        
        # No matches found - provide smart fallback
        return {
            "response": {
                "type": "no_results",
                "message": "I couldn't find any products matching your search. Here are some popular items you might like:",
                "products": [self.format_product_details(p) for p in self.get_popular_products()[:3]],
                "suggestions": [
                    "Browse all categories",
                    "Try different keywords",
                    "See bestsellers"
                ]
            }
        }

    def personalize_results(self, products, preferences):
        """Personalize product results based on user preferences"""
        scored_products = []
        for product in products:
            score = 0
            
            # Category preference
            if preferences.get('preferred_category') == product['category']:
                score += 50
                
            # Price range preference
            if preferences.get('price_range'):
                min_price, max_price = preferences['price_range']
                if min_price <= product['price'] <= max_price:
                    score += 30
                    
            # Rating preference
            if product.get('rating', 0) >= preferences.get('min_rating', 0):
                score += 20
                
            scored_products.append((score, product))
            
        # Sort by score and return products
        scored_products.sort(reverse=True, key=lambda x: x[0])
        return [p for _, p in scored_products]

    def handle_product_comparison(self, query, session_id):
        """Enhanced product comparison with personalization"""
        # Find matching products
        matches = self.find_matching_products(query)
        
        # Apply personalization
        if session_id and session_id in self.user_preferences:
            matches = self.personalize_results(matches, self.user_preferences[session_id])
        
        if matches:
            return {
                "response": {
                    "type": "product_comparison",
                    "message": f"Comparing {len(matches)} products:",
                    "products": [self.format_product_details(p) for p in matches[:3]],
                    "comparison": self.generate_product_comparison(matches[:3]),
                    "suggestions": [
                        "Show more details",
                        "Compare prices",
                        "Show reviews"
                    ]
                }
            }
        
        return self.get_smart_fallback_response(query)

    def handle_recommendation(self, query, session_id):
        """Generate personalized product recommendations"""
        # Get user preferences and history
        user_prefs = self.user_preferences.get(session_id, {})
        history = self.conversation_history.get(session_id, [])
        
        # Find matching products
        matches = self.find_matching_products(query)
        if not matches:
            return self.get_fallback_response()
            
        # Sort by recommendation score
        scored_products = [
            (p, self.calculate_recommendation_score(p, user_prefs, history))
            for p in matches
        ]
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "response": {
                "type": "recommendations",
                "best_match": self.format_product_details(scored_products[0][0]),
                "alternatives": [
                    self.format_product_details(p) 
                    for p, _ in scored_products[1:3]
                ],
                "explanation": self.generate_recommendation_explanation(
                    scored_products[0][0],
                    user_prefs
                ),
                "suggestions": [
                    "Tell me more",
                    "Show reviews",
                    "Compare alternatives"
                ]
            }
        }

    def calculate_recommendation_score(self, product, user_prefs, history):
        """Calculate personalized recommendation score"""
        score = 0
        
        # Base score from product attributes
        score += product.get('rating', 0) * 20
        score += min(product.get('reviews_count', 0), 100)
        
        # Preference matching
        if user_prefs.get('preferred_category') == product['category']:
            score += 50
            
        # Price preference
        if user_prefs.get('price_range'):
            min_price, max_price = user_prefs['price_range']
            if min_price <= product['price'] <= max_price:
                score += 30
                
        # Recent interaction bonus
        if history:
            recent_queries = [h['query'].lower() for h in history[-5:]]
            if any(word in ' '.join(recent_queries) for word in product['keywords']):
                score += 40
                
        return score

    def handle_order_tracking(self, query, session_id):
        """Handle order tracking queries"""
        # Check for specific order tracking keywords
        order_tracking_keywords = ['track', 'delivery', 'package']
        if any(keyword in query.lower() for keyword in order_tracking_keywords):
            return {
                "response": {
                    "type": "order_tracking",
                    "message": "I'm sorry, but I can't directly track orders. I recommend checking your order status through your account or contacting our customer service for assistance.",
                    "suggestions": ["Check order status", "Contact support"]
                }
            }
        
        return self.get_smart_fallback_response(query)

    def handle_customer_service(self, query):
        """Handle customer service queries"""
        # Check for common customer service topics
        customer_service_topics = {
            "delivery": "delivery time|shipping|when.*arrive",
            "warranty": "warranty|guarantee|broken",
            "payment": "pay|price|cost|discount",
            "return": "return|refund|exchange",
            "technical": "how.*use|manual|instructions"
        }
        
        for topic, patterns in customer_service_topics.items():
            if any(p in query.lower() for p in patterns.split("|")):
                return {
                    "response": {
                        "type": "customer_service",
                        "topic": topic,
                        "message": self.product_data['support_info'].get(topic, ""),
                        "contact_info": self.product_data['support_info']['contact'],
                        "suggestions": ["Need more help?", "Show products", "Contact support"]
                    }
                }
        
        return self.generate_smart_response(query)

    def generate_conversational_response(self, query, session_id):
        """Generate conversational response for general queries"""
        try:
            # First check if it's a product query
            if any(word in query.lower() for word in ["show", "find", "recommend", "best", "product", "tool"]):
                matches = self.find_matching_products(query)
                if matches:
                    return {
                        "response": {
                            "type": "product_recommendations",
                            "message": "Based on your interest, here are some recommendations:",
                            "products": [self.format_product_details(p) for p in matches[:3]],
                            "explanation": self.generate_recommendation_explanation(matches[0]),
                            "suggestions": [
                                "Compare these products",
                                "Show more details",
                                "See reviews"
                            ]
                        }
                    }

            # Check for common questions
            for topic, answer in self.product_data['faqs'].items():
                if topic.lower() in query.lower():
                    return {
                        "response": {
                            "type": "faq",
                            "message": answer,
                            "related_products": self.find_related_products(topic),
                            "suggestions": [
                                "Need more help?",
                                "Browse products",
                                "Contact support"
                            ]
                        }
                    }

            # Generate conversational response
            context = f"""
            Customer: {query}
            AI Assistant: I'm here to help you find the perfect tools and answer any questions about our products.
            """
            
            inputs = self.gpt2_tokenizer(context, return_tensors="pt", max_length=512)
            outputs = self.gpt2_model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                num_return_sequences=1
            )
            
            response = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "response": {
                    "type": "conversational",
                    "message": response,
                    "suggestions": [
                        "Show popular products",
                        "Browse categories",
                        "Special offers"
                    ]
                }
            }
            
        except Exception as e:
            print(f"Error generating conversational response: {str(e)}")
            return self.get_fallback_response()

    def translate_response(self, response, target_lang):
        """Translate the response to target language"""
        try:
            if response.get('response', {}).get('message'):
                translated_message = self.translator.translate(
                    response['response']['message'], 
                    dest=target_lang
                ).text
                response['response']['message'] = translated_message
            
            # Translate suggestions
            if response.get('response', {}).get('suggestions'):
                translated_suggestions = [
                    self.translator.translate(s, dest=target_lang).text 
                    for s in response['response']['suggestions']
                ]
                response['response']['suggestions'] = translated_suggestions
            
            # Translate product information
            if response.get('response', {}).get('products'):
                for product in response['response']['products']:
                    product['name'] = self.translator.translate(
                        product['name'], 
                        dest=target_lang
                    ).text
                    product['description'] = self.translator.translate(
                        product['description'], 
                        dest=target_lang
                    ).text
            
            return response
            
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return response

    def generate_contextual_response(self, query, conversation):
        """Generate response based on conversation context"""
        try:
            # Analyze conversation context
            context = self.analyze_conversation_context(conversation)
            
            # Get query intent
            intent = self.analyze_query_intent(query)
            
            # Handle based on context and intent
            if context.get('previous_products'):
                # User is discussing specific products
                return self.handle_product_followup(query, context)
            elif context.get('support_topic'):
                # User is in middle of support conversation
                return self.handle_support_followup(query, context)
            elif context.get('comparing_products'):
                # User is comparing products
                return self.handle_comparison_followup(query, context)
            else:
                # New conversation thread
                return self.handle_new_query(query, intent)
                
        except Exception as e:
            print(f"Error generating contextual response: {str(e)}")
            return self.get_fallback_response()

    def analyze_conversation_context(self, conversation):
        """Analyze conversation history for context"""
        context = {
            'previous_products': [],
            'support_topic': None,
            'comparing_products': False,
            'user_preferences': {},
            'last_intent': None
        }
        
        if not conversation:
            return context
            
        # Analyze last 3 interactions
        recent = conversation[-3:]
        for interaction in recent:
            if 'products' in interaction.get('bot', {}).get('response', {}):
                context['previous_products'].extend(
                    interaction['bot']['response']['products']
                )
            if interaction.get('bot', {}).get('response', {}).get('type') == 'support':
                context['support_topic'] = interaction['bot']['response'].get('topic')
            if 'compare' in interaction['user'].lower():
                context['comparing_products'] = True
                
        return context

    def analyze_query_intent(self, query):
        """Enhanced intent analysis"""
        query_embedding = self.nlp_model.encode(query.lower(), convert_to_tensor=True)
        
        # Match against training examples
        best_match = None
        best_score = 0
        
        for example in self.product_data['training_examples']:
            score = util.pytorch_cos_sim(
                query_embedding,
                self._generate_training_embeddings()[example['query']]
            ).item()
            if score > best_score:
                best_score = score
                best_match = example
        
        # Determine intent type
        intent_types = {
            'product': ['show', 'find', 'buy', 'price', 'compare'],
            'order': ['order', 'track', 'delivery', 'shipping'],
            'support': ['help', 'issue', 'problem', 'broken'],
            'account': ['login', 'account', 'password'],
            'refund': ['refund', 'return', 'money back']
        }
        
        query_lower = query.lower()
        for intent_type, keywords in intent_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return {
                    'type': intent_type,
                    'confidence': best_score,
                    'matched_example': best_match,
                    'requires_auth': intent_type in ['order', 'account', 'refund']
                }
        
        return {'type': 'general', 'confidence': best_score}

    def handle_product_query(self, query, user_id):
        """Enhanced product query handling"""
        # Get user preferences
        preferences = self.user_preferences.get(user_id, {})
        
        # Find matching products with personalization
        matches = self.find_matching_products(query)
        if preferences.get('preferred_category'):
            matches = [p for p in matches if p['category'] == preferences['preferred_category']] + matches
        
        if not matches:
            return self.get_smart_fallback_response(query)
        
        # Generate personalized response
        response = {
            'type': 'product_recommendation',
            'products': matches[:3],
            'explanation': self.generate_product_explanation(matches[0], user_id),
            'comparison': self.generate_product_comparison(matches[:3]) if len(matches) > 1 else None,
            'related_products': self.find_related_products(matches[0]),
            'user_specific': {
                'previous_purchases': self.get_user_purchases(user_id),
                'recommended_accessories': self.get_recommended_accessories(matches[0])
            }
        }
        
        return response

    def handle_support_query(self, query, user_id):
        """Enhanced support query handling"""
        # Check if product-specific issue
        product = self.find_product_in_query(query)
        
        if product:
            return {
                'type': 'technical_support',
                'product': product,
                'troubleshooting_steps': self.get_troubleshooting_steps(product),
                'warranty_info': product.get('offers', {}).get('warranty'),
                'support_options': self.get_support_options(product),
                'related_docs': self.get_product_documentation(product),
                'video_tutorials': self.get_video_tutorials(product)
            }
        
        # General support query
        return {
            'type': 'general_support',
            'faq_matches': self.find_matching_faqs(query),
            'contact_options': self.product_data['support_info']['contact'],
            'suggested_actions': self.get_suggested_actions(query)
        }

    def needs_human_handover(self, query, response):
        """Determine if query needs human attention"""
        conditions = [
            response.get('confidence', 1.0) < 0.4,
            'urgent' in query.lower(),
            'speak to human' in query.lower(),
            response.get('type') == 'technical_support' and response.get('severity') == 'high'
        ]
        return any(conditions)

    def prepare_human_handover(self, response):
        """Prepare response for human handover"""
        response.update({
            'handover': True,
            'queue_position': self.get_support_queue_position(),
            'estimated_wait': '5-10 minutes',
            'support_hours': self.product_data['support_info']['contact']['hours'],
            'alternative_contact': {
                'email': self.product_data['support_info']['contact']['email'],
                'phone': self.product_data['support_info']['contact']['phone']
            }
        })
        return response

    def process_query(self, user_input):
        """Main method to process user queries"""
        try:
            # Add query to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Analyze query intent
            intent = self.analyze_query_intent(user_input)
            
            if intent["type"] == "product":
                return self.handle_product_query(user_input, intent)
            elif intent["type"] == "general":
                return self.handle_general_query(user_input, intent)
            else:
                return self.get_smart_fallback_response(user_input)
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return self.get_fallback_response()

    def analyze_query(self, query):
        # Identify query intent and type
        query_lower = query.lower()
        intents = {
            'compare': any(word in query_lower for word in ['compare', 'difference', 'better']),
            'recommend': any(word in query_lower for word in ['recommend', 'suggest', 'best']),
            'price': any(word in query_lower for word in ['price', 'cost', 'expensive', 'cheap']),
            'features': any(word in query_lower for word in ['features', 'specifications', 'specs']),
            'reviews': any(word in query_lower for word in ['reviews', 'ratings', 'feedback'])
        }
        return intents

    def get_product_response(self, user_input):
        try:
            query_embedding = self.nlp_model.encode(user_input.lower(), convert_to_tensor=True)
            intents = self.analyze_query(user_input)
            
            # Find best matching products
            matches = []
            for product in self.product_data["products"]:
                similarity = util.pytorch_cos_sim(
                    query_embedding, 
                    self._generate_product_embeddings()[product['id']]
                ).item()
                if similarity > 0.3:  # Adjust threshold as needed
                    matches.append((similarity, product))
            
            matches.sort(reverse=True, key=lambda x: x[0])
            
            if matches:
                if intents['compare'] and len(matches) >= 2:
                    return self.generate_comparison(matches[:2])
                elif intents['recommend']:
                    return self.generate_recommendation(matches)
                elif intents['price']:
                    return self.generate_price_info(matches)
                elif intents['features']:
                    return self.generate_feature_details(matches[0][1])
                elif intents['reviews']:
                    return self.generate_review_summary(matches[0][1])
                else:
                    return self.generate_product_details(matches[0][1])
            
            return self.get_fallback_response()
            
        except Exception as e:
            print(f"Error in get_product_response: {str(e)}")
            return self.get_fallback_response()

    def generate_product_details(self, product):
        return {
            "response": {
                "type": "product_detail",
                "product": {
                    "id": product["id"],
                    "name": product["name"],
                    "description": product["description"],
                    "price": product["price"],
                    "original_price": product.get("original_price"),
                    "features": product["features"],
                    "rating": product.get("rating", 0),
                    "offers": product.get("offers", {}),
                    "feedback_summary": self._summarize_feedback(product)
                }
            },
            "message": f"Here's what I found about the {product['name']}:",
            "suggestions": [
                "Would you like to see similar products?",
                "Compare with other models",
                "Show me the reviews",
                "Tell me about the warranty"
            ]
        }

    def _summarize_feedback(self, product):
        if "feedback" not in product:
            return "No feedback available yet"
        
        total_ratings = len(product["feedback"])
        avg_rating = sum(f["rating"] for f in product["feedback"]) / total_ratings
        
        return f"Average rating: {avg_rating:.1f}/5 based on {total_ratings} reviews"

    def get_product_id(self, user_input):
        try:
            if not self.model_loaded:
                return self.get_fallback_response()

            user_input = user_input.lower()
            self.conversation_history.append(user_input)

            # Identify intent
            if "compare" in user_input or "difference" in user_input:
                return self.compare_products(user_input)
            elif "recommend" in user_input or "suggest" in user_input or "best" in user_input:
                return self.recommend_products(user_input)
            elif "price" in user_input or "cost" in user_input:
                return self.get_price_info(user_input)
            else:
                return self.search_products(user_input)

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return self.get_fallback_response()

    def search_products(self, query):
        try:
            search_terms = query.lower().split()
            
            # Handle general product queries
            if any(word in query.lower() for word in ["show", "all", "products", "tools"]):
                # If asking for all products or tools
                if "power" in query.lower():
                    return self.get_category_products("power-tools")
                elif "hand" in query.lower():
                    return self.get_category_products("hand-tools")
                else:
                    # Show all products
                    return self.get_category_products()

            # Regular search
            best_matches = []
            for product in self.product_data["products"]:
                search_text = f"{product['name']} {product['description']} {' '.join(product['keywords'])} {product['category']}"
                search_text = search_text.lower()
                
                # More flexible matching
                match_score = 0
                for term in search_terms:
                    if term in search_text:
                        match_score += 1
                    # Check keywords specifically
                    if any(term in keyword.lower() for keyword in product['keywords']):
                        match_score += 2  # Give extra weight to keyword matches
                
                if match_score > 0:
                    best_matches.append((match_score, product))

            if best_matches:
                # Sort by match score
                best_matches.sort(reverse=True, key=lambda x: x[0])
                matched_products = [product for score, product in best_matches]
                
                return {
                    "response": {
                        "products": [
                            {
                                "id": product["id"],
                                "name": product["name"],
                                "description": product["description"],
                                "price": product["price"],
                                "category": product["category"],
                                "features": [
                                    f"Category: {product['category']}",
                                    f"Brand: {product['name'].split()[0]}",
                                    "In Stock"
                                ]
                            } for product in matched_products
                        ],
                        "type": "product_list",
                        "total_found": len(matched_products)
                    },
                    "message": f"Found {len(matched_products)} products for you:",
                    "suggestions": [
                        "Show all power tools",
                        "Show all hand tools",
                        f"Show me {matched_products[0]['category']} products",
                        "Sort by price"
                    ]
                }
            
            # If no matches found, show categories
            return self.get_category_suggestions()

        except Exception as e:
            print(f"Search error: {str(e)}")
            return self.get_fallback_response()

    def get_category_products(self, category=None):
        try:
            products = self.product_data["products"]
            if category:
                products = [p for p in products if p["category"] == category]
            
            return {
                "response": {
                    "products": [
                        {
                            "id": product["id"],
                            "name": product["name"],
                            "description": product["description"],
                            "price": product["price"],
                            "category": product["category"],
                            "features": [
                                f"Category: {product['category']}",
                                f"Brand: {product['name'].split()[0]}",
                                "In Stock"
                            ]
                        } for product in products
                    ],
                    "type": "category_list",
                    "category": category or "all tools",
                    "total_found": len(products)
                },
                "message": f"Showing all {category or 'tools'} ({len(products)} items):",
                "suggestions": [
                    "Filter by brand",
                    "Sort by price",
                    "Show bestsellers only",
                    "Compare similar items"
                ]
            }
        except Exception as e:
            print(f"Category error: {str(e)}")
            return self.get_fallback_response()

    def get_category_suggestions(self):
        return {
            "response": {
                "message": "Here are our product categories:",
                "categories": [
                    {
                        "name": "Power Tools",
                        "count": len([p for p in self.product_data["products"] if p["category"] == "power-tools"]),
                        "examples": ["Drills", "Saws", "Nailers"]
                    },
                    {
                        "name": "Hand Tools",
                        "count": len([p for p in self.product_data["products"] if p["category"] == "hand-tools"]),
                        "examples": ["Spanners", "Hammers", "Screwdrivers"]
                    }
                ],
                "type": "category_suggestions"
            },
            "suggestions": [
                "Show all power tools",
                "Show all hand tools",
                "What's new in stock?",
                "Show bestsellers"
            ]
        }

    def recommend_products(self, query):
        category = "power-tools"  # Default category
        if "hand" in query:
            category = "hand-tools"
        
        matching_products = [p for p in self.product_data["products"] if p["category"] == category]
        if matching_products:
            sorted_products = sorted(matching_products, key=lambda x: x["price"])
            return {
                "response": {
                    "products": [
                        {
                            "id": product["id"],
                            "name": product["name"],
                            "description": product["description"],
                            "price": product["price"],
                            "category": product["category"],
                            "recommendation_reason": "Best value for money" if idx == 0 else "Popular choice" if idx == 1 else "Premium option"
                        } for idx, product in enumerate(sorted_products[:3])
                    ],
                    "type": "recommendation"
                },
                "message": f"Here are my top recommendations in {category}:"
            }
        return self.get_fallback_response()

    def get_fallback_response(self, lang='en'):
        """Get fallback response in appropriate language"""
        fallback_messages = {
            'en': "I'm sorry, I didn't understand that. Could you please rephrase?",
            'ta': "மன்னிக்கவும், எனக்கு புரியவில்லை. தயவுசெய்து மீண்டும் கூறுங்கள்?",
            # Add more languages as needed
        }
        
        return {
            "response": {
                "type": "fallback",
                "message": fallback_messages.get(lang, fallback_messages['en']),
                "suggestions": self.get_suggestions_in_language(lang)
            }
        }

    def get_suggestions_in_language(self, lang):
        """Get suggestions in appropriate language"""
        suggestions = {
            'en': [
                "Show all products",
                "Contact support",
                "Browse categories"
            ],
            'ta': [
                "அனைத்து தயாரிப்புகளையும் காட்டு",
                "ஆதரவை தொடர்பு கொள்ளவும்",
                "வகைகளை உலாவ"
            ]
            # Add more languages as needed
        }
        return suggestions.get(lang, suggestions['en'])

    def basic_search(self, query):
        try:
            print(f"Performing basic search for: {query}")
            query = query.lower()
            matches = []
            
            # First, analyze the query intent
            is_question = any(word in query for word in ["what", "which", "how", "can", "do", "where", "when"])
            is_product_query = any(word in query for word in ["show", "find", "looking", "want", "need", "search"])
            
            # Handle different types of queries
            if is_question:
                return self.handle_question(query)
            elif is_product_query:
                matches = self.find_matching_products(query)
            else:
                # General product search
                matches = self.find_matching_products(query)

            if matches:
                return {
                    "response": {
                        "type": "interactive_product_list",
                        "products": [self.format_product_details(p) for p in matches],
                        "total_found": len(matches),
                        "question": "Would you like to see more details about these products?",
                        "options": ["Yes", "No"],
                        "follow_up": {
                            "Yes": "show_detailed_comparison",
                            "No": "show_alternative_suggestions"
                        }
                    },
                    "message": f"I found {len(matches)} products that might interest you:",
                    "suggestions": [
                        "Compare prices",
                        "Show reviews",
                        "View specifications",
                        "See similar products"
                    ]
                }
            
            return self.get_smart_fallback_response(query)
            
        except Exception as e:
            print(f"Error in basic search: {str(e)}")
            return self.get_fallback_response()

    def format_product_details(self, product):
        return {
            "id": product["id"],
            "name": product["name"],
            "description": product["description"],
            "price": product["price"],
            "category": product["category"],
            "features": product.get("features", []),
            "rating": product.get("rating", 0),
            "reviews_count": product.get("reviews_count", 0),
            "offers": product.get("offers", {}),
            "comparison_points": [
                f"Price: ${product['price']}",
                f"Rating: {product.get('rating', 0)}/5",
                f"Category: {product['category']}",
                product.get('offers', {}).get('warranty', 'Standard warranty')
            ],
            "recommendation_score": self.calculate_recommendation_score(product)
        }

    def handle_question(self, query):
        # Handle general questions
        if "price" in query or "cost" in query:
            return self.get_price_information(query)
        elif "review" in query or "rating" in query:
            return self.get_product_reviews(query)
        elif "compare" in query or "difference" in query:
            return self.compare_products(query)
        elif "best" in query or "recommend" in query:
            return self.get_best_recommendations(query)
        else:
            return self.get_general_response(query)

    def get_best_recommendations(self, query):
        matches = self.find_matching_products(query)
        if not matches:
            return self.get_fallback_response()

        # Sort by recommendation score
        sorted_products = sorted(
            matches,
            key=lambda p: self.calculate_recommendation_score(p),
            reverse=True
        )

        return {
            "response": {
                "type": "recommendation",
                "best_match": self.format_product_details(sorted_products[0]),
                "alternatives": [self.format_product_details(p) for p in sorted_products[1:4]],
                "explanation": self.generate_recommendation_explanation(sorted_products[0]),
                "question": "Would you like to learn more about this product?",
                "options": ["Yes", "Show alternatives", "No"]
            },
            "message": "Based on your requirements, I recommend:",
            "suggestions": [
                "Compare with alternatives",
                "Show reviews",
                "View specifications",
                "See price details"
            ]
        }

    def calculate_recommendation_score(self, product):
        score = 0
        score += product.get('rating', 0) * 20  # Up to 100 points for rating
        score += min(product.get('reviews_count', 0), 100)  # Up to 100 points for review count
        if product.get('offers', {}).get('discount'):
            score += 50  # Bonus for discounted items
        return score

    def generate_recommendation_explanation(self, product):
        reasons = []
        if product.get('rating', 0) >= 4:
            reasons.append(f"Highly rated ({product['rating']}/5 stars)")
        if product.get('reviews_count', 0) > 50:
            reasons.append(f"Well reviewed by {product['reviews_count']} customers")
        if product.get('offers', {}).get('discount'):
            reasons.append(f"Currently {product['offers']['discount']}% off")
        if product.get('offers', {}).get('warranty'):
            reasons.append(f"Includes {product['offers']['warranty']}")
        
        return {
            "title": f"Why I recommend the {product['name']}:",
            "reasons": reasons,
            "key_features": product.get('features', [])[:3]  # Top 3 features
        }

    def find_matching_products(self, query):
        """Find products matching the query"""
        matches = []
        query = query.lower()
        
        # Handle category queries
        if "all" in query or "show" in query:
            if "power" in query:
                return [p for p in self.product_data["products"] if p["category"] == "power-tools"]
            elif "hand" in query:
                return [p for p in self.product_data["products"] if p["category"] == "hand-tools"]
            else:
                return self.product_data["products"]

        # Search in product details
        for product in self.product_data["products"]:
            search_text = f"{product['name']} {product['description']} {' '.join(product['keywords'])}".lower()
            if any(term in search_text for term in query.split()):
                matches.append(product)
        
        return matches

    def get_price_information(self, query):
        """Get price information for products"""
        matches = self.find_matching_products(query)
        if not matches:
            return self.get_fallback_response()
            
        return {
            "response": {
                "type": "price_info",
                "products": [self.format_product_details(p) for p in matches],
                "price_range": {
                    "min": min(p["price"] for p in matches),
                    "max": max(p["price"] for p in matches)
                },
                "question": "Would you like to see products in a specific price range?",
                "options": ["Show cheapest", "Show all", "Show premium"]
            },
            "message": "Here are the price details you requested:"
        }

    def get_product_reviews(self, query):
        """Get product reviews"""
        matches = self.find_matching_products(query)
        if not matches:
            return self.get_fallback_response()
            
        return {
            "response": {
                "type": "reviews",
                "products": [{
                    **self.format_product_details(p),
                    "reviews": p.get("feedback", [])
                } for p in matches],
                "question": "Would you like to see more detailed reviews?",
                "options": ["Yes", "No"]
            },
            "message": "Here are the product reviews:"
        }

    def get_general_response(self, query):
        """Handle general questions"""
        general_responses = {
            "delivery": "We offer free delivery on orders over $100. Standard delivery takes 3-5 business days.",
            "warranty": "All our power tools come with a minimum 1-year warranty. Some products have extended warranty options.",
            "return": "You can return unused products within 30 days for a full refund.",
            "payment": "We accept all major credit cards, PayPal, and Apple Pay.",
            "store": "Our physical store is located at 123 Tool Street. We're open Mon-Sat, 9AM-6PM."
        }

        # Find the most relevant response
        for key, response in general_responses.items():
            if key in query.lower():
                return {
                    "response": {
                        "type": "general_info",
                        "message": response,
                        "question": "Is there anything else you'd like to know?",
                        "options": ["Yes", "No"]
                    },
                    "message": "Here's what you need to know:"
                }

        return self.get_smart_fallback_response(query)

    def get_smart_fallback_response(self, query):
        """Provide a smart fallback response with suggestions"""
        return {
            "response": {
                "type": "suggestions",
                "message": "I'm not sure I understood your question. Here are some suggestions:",
                "suggestions": [
                    "Show all power tools",
                    "What's on sale?",
                    "Compare popular drills",
                    "Show best-rated products"
                ],
                "categories": [
                    {"name": "Power Tools", "count": len([p for p in self.product_data["products"] if p["category"] == "power-tools"])},
                    {"name": "Hand Tools", "count": len([p for p in self.product_data["products"] if p["category"] == "hand-tools"])}
                ]
            },
            "message": "I couldn't find exactly what you're looking for, but here are some options:"
        }

    def handle_user_choice(self, choice, previous_response):
        """Handle user's choice from previous interaction"""
        if not previous_response or "type" not in previous_response.get("response", {}):
            return self.get_fallback_response()

        response_type = previous_response["response"]["type"]
        
        if response_type == "interactive_product_list":
            if choice == "Yes":
                return self.show_detailed_comparison(previous_response["response"]["products"])
            else:
                return self.show_alternative_suggestions()
                
        elif response_type == "recommendation":
            if choice == "Yes":
                return self.show_product_details(previous_response["response"]["best_match"])
            elif choice == "Show alternatives":
                return self.show_alternatives(previous_response["response"]["alternatives"])
                
        return self.get_fallback_response()

    def find_related_products(self, topic_or_product):
        """Find related products based on topic or product"""
        try:
            if isinstance(topic_or_product, str):
                # Find products related to topic
                query_embedding = self.nlp_model.encode(topic_or_product, convert_to_tensor=True)
                matches = []
                for product in self.product_data["products"]:
                    similarity = util.pytorch_cos_sim(
                        query_embedding,
                        self.product_embeddings[product['id']]
                    ).item()
                    if similarity > 0.3:
                        matches.append((similarity, product))
                matches.sort(reverse=True, key=lambda x: x[0])
                return [m[1] for m in matches[:3]]
            else:
                # Find products related to a product
                category = topic_or_product.get('category')
                price_range = (
                    topic_or_product['price'] * 0.8,
                    topic_or_product['price'] * 1.2
                )
                return [
                    p for p in self.product_data["products"]
                    if p['category'] == category 
                    and price_range[0] <= p['price'] <= price_range[1]
                    and p['id'] != topic_or_product['id']
                ][:3]
        except Exception as e:
            print(f"Error finding related products: {str(e)}")
            return []

# Initialize chatbot
chatbot = None

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        user_input = data.get('user_input', '')
        session_id = data.get('session_id')
        
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
        
        # Initialize chatbot if needed
        global chatbot
        if not chatbot:
            chatbot = ProductAI()
            print("Initialized new ProductAI instance")

        # Process the query
        try:
            # First try exact matches from training data
            for example in chatbot.product_data['training_examples']:
                if user_input.lower() in example['query'].lower():
                    return jsonify({
                        "response": {
                            "type": "trained_response",
                            "message": example['response'],
                            "suggestions": [
                                "Tell me more",
                                "Show similar products",
                                "Compare prices"
                            ]
                        }
                    })

            # Check for product-related queries
            if any(word in user_input.lower() for word in ["show", "find", "product", "tool", "price"]):
                matches = chatbot.find_matching_products(user_input)
                if matches:
                    return jsonify({
                        "response": {
                            "type": "product_list",
                            "message": f"I found {len(matches)} products that might interest you:",
                            "products": [chatbot.format_product_details(p) for p in matches[:3]],
                            "suggestions": ["Compare prices", "Show reviews", "More details"]
                        }
                    })

            # Check FAQs
            for topic, answer in chatbot.product_data['faqs'].items():
                if topic.lower() in user_input.lower():
                    return jsonify({
                        "response": {
                            "type": "faq",
                            "message": answer,
                            "suggestions": ["Need more help?", "Show products", "Contact support"]
                        }
                    })

            # Generate AI response for other queries
            response = chatbot.process_customer_query(user_input, session_id)
            return jsonify(response)

        except Exception as e:
            print(f"Query processing error: {str(e)}")
            return jsonify({
                "response": {
                    "type": "error",
                    "message": "I'm having trouble understanding. Could you rephrase that?",
                    "suggestions": ["Show products", "Need help?", "Contact support"]
                }
            })
            
    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({
            "response": {
                "type": "error",
                "message": "Sorry, I encountered an error. Please try again."
            }
        }), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename, mimetype='image/jpeg')

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Access the website at http://localhost:8000")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.get_device_name()}")
    app.run(debug=True, port=8000)