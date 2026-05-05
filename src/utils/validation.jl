# ============================================================
# 输入校验
# ============================================================

function validate_user_id(relation::DirectRelation, user_id::Int)
    user_id in relation.user_ids ||
        throw(ArgumentError("用户 $user_id 不存在"))
end

function validate_item_id(relation::DirectRelation, item_id::Int)
    item_id in relation.item_ids ||
        throw(ArgumentError("物品 $item_id 不存在"))
end
