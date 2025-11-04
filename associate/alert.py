import requests
from typing import Optional, List
from components.logger import setup_logger
from components.event_logger import event_logger
from entries.track_entry import AssociateResult
from source import get_watch_client

logger = setup_logger("Alert")

def risk_alert(message: list[AssociateResult], event_id: Optional[str] = None, track_ids: Optional[List[str]] = None) -> None:
    """
    溺水告警推送：
    - 输入为本批次的检测结果（同一事件ID）
    - 对 `get_watch_client()` 返回的每个客户端进行推送
    - 逐个记录推送结果，并在末尾写入事件级汇总（finish）
    """
    warnList = [result.to_dict() for result in message]
    logger.info(f"[EVENT-START] event_id={event_id} tracks={len(warnList)} track_ids={track_ids}")
    event_logger.log(
        'start',
        event_id=event_id,
        track_ids=track_ids,
        count=len(warnList),
    )
    ok, fail = 0, 0

    for client in get_watch_client():
        # 准备要发送的数据
        data = {
            "cid": client,
            "payload": {
                "warnList": warnList,
                "eventId": event_id,
                "trackIds": track_ids,
            },
            "title": "告警通知",
            "content": "溺水告警通知"
        }

        # 设置请求头
        headers = {
            'Content-Type': 'application/json',
        }

        # 发起POST请求
        try:
            # 推送接口：如有替换需求，请保持 payload 字段名不变（外部系统依赖）
            response = requests.post(
                url='https://fc-mp-1aedb5cf-94f8-410b-baaf-284858354836.next.bspapp.com/push2',
                json=data,
                headers=headers,
                timeout=10
            )

            response.raise_for_status()
            result = response.json()
            logger.info(f"[PUSH-OK] event_id={event_id} client={client} result={result}")
            event_logger.log('push_ok', event_id=event_id, client=client, result=result)
            ok += 1

        except requests.exceptions.RequestException as e:
            logger.warning(f"[PUSH-FAIL] event_id={event_id} client={client} error={e}")
            event_logger.log('push_fail', event_id=event_id, client=client, error=str(e))
            fail += 1
    # 事件结束
    logger.info(f"[EVENT-END] event_id={event_id} ok={ok} fail={fail}")
    event_logger.log('finish', event_id=event_id, ok=ok, fail=fail)
